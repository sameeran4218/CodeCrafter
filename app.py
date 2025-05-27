import streamlit as st
import os
import json
import time
import re
import requests
from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

# LangChain imports
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema.runnable import RunnableLambda, Runnable
from boltiotai import openai

OPENAI_API_KEY = openai.api_key = os.getenv('OPENAI_API_KEY')
HUGGINGFACE_TOKEN=os.getenv('HUGGINGFACE_TOKEN')


class ProblemAnalysis(BaseModel):
    problem_name: str = Field(description="Name of the coding problem")
    input_format: List[Dict[str, str]] = Field(description="List of input parameters with name and type")
    output_format: Dict[str, str] = Field(description="Output format with type and description")
    constraints: List[str] = Field(description="List of problem constraints")
    logic_summary: str = Field(description="Detailed explanation of the solution approach")
    dsa_topics: List[str] = Field(description="Relevant data structures and algorithms topics")
    edge_cases: List[str] = Field(description="Important edge cases to consider")


class LightweightMemory:
    def __init__(self, max_pairs: int = 3):
        self.max_pairs = max_pairs
        self.qa_pairs: List[Dict] = []
        self.current_session_topics = set()

    def add_qa_pair(self, problem_name: str, topic: str, user_msg: str, assistant_msg: str):
        compressed_pair = {
            "problem": problem_name,
            "topic": topic,
            "user": user_msg[:200] + "..." if len(user_msg) > 200 else user_msg,
            "solution_summary": self._extract_solution_summary(assistant_msg),
            "timestamp": datetime.now().strftime("%H:%M")
        }
        self.qa_pairs.append(compressed_pair)
        self.current_session_topics.add(topic)
        if len(self.qa_pairs) > self.max_pairs:
            self.qa_pairs.pop(0)

    def _extract_solution_summary(self, solution: str) -> str:
        lines = solution.split('\n')
        summary_parts = []
        for line in lines:
            if any(word in line.lower() for word in ['time:', 'space:', 'complexity', 'o(']):
                summary_parts.append(line.strip())
        for line in lines[:5]:
            if '//' in line or '#' in line:
                summary_parts.append(line.strip()[:50])
                break
        return ' | '.join(summary_parts[:2]) if summary_parts else "Solution provided"

    def get_compact_context(self) -> str:
        if not self.qa_pairs:
            return ""
        context_parts = []
        context_parts.append(f"Session topics: {', '.join(list(self.current_session_topics)[-3:])}")
        if self.qa_pairs:
            recent = self.qa_pairs[-1]
            context_parts.append(f"Last: {recent['problem']} ({recent['topic']})")
        return " | ".join(context_parts)

    def clear(self):
        self.qa_pairs.clear()
        self.current_session_topics.clear()


# Agent 1: Problem Analysis (HuggingFace Direct API)
class ProblemAnalyzer:
    def __init__(self):
        self.token = HUGGINGFACE_TOKEN
        self.headers = {"Authorization": f"Bearer {self.token}"}
        # Using Mistral-7B-Instruct which is better for structured tasks
        self.api_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
        self._setup_prompts()

    def _setup_prompts(self):
        example_prompt = ChatPromptTemplate.from_messages([
            ("user", "Question:\n{input}"),
            ("ai", "Answer:\n{output}")
        ])

        examples = [
            {
                "input": "Given an array of integers nums and an integer k, return the number of contiguous subarrays whose sum is equal to k.",
                "output": json.dumps({
                    "problem_name": "Subarray Sum Equals K",
                    "input_format": [{"name": "nums", "type": "List[int]"}, {"name": "k", "type": "int"}],
                    "output_format": {"type": "int", "description": "Number of subarrays whose sum equals k"},
                    "constraints": ["1 <= nums.length <= 10^5", "-10^4 <= nums[i] <= 10^4", "-10^7 <= k <= 10^7"],
                    "logic_summary": "To solve this efficiently, we maintain a running prefix sum while iterating through the array and use a hashmap to keep track of how many times each prefix sum has occurred. At each index, we check if (current_sum - k) exists in the hashmap, which would mean there's a subarray ending at the current index that sums to k, and we increment our count by the number of times that sum has appeared.",
                    "dsa_topics": ["Prefix Sum", "Hash Map"],
                    "edge_cases": ["All elements are 0", "No subarray adds up to k"]
                })
            }
        ]

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        system_prompt = (
            "You are a coding problem analyzer. You MUST respond with ONLY valid JSON in this exact format:\n\n"
            '''{{
  "problem_name": "string",
  "input_format": [{{"name": "string", "type": "string"}}],
  "output_format": {{"type": "string", "description": "string"}},
  "constraints": ["string"],
  "logic_summary": "string",
  "dsa_topics": ["string"],
  "edge_cases": ["string"]
}}'''
            "\n\nIMPORTANT RULES:\n"
            "- Respond with ONLY the JSON object\n"
            "- Do NOT add any text before or after the JSON\n"
            "- Do NOT use markdown formatting\n"
            "- Ensure all strings are properly quoted\n"
            "- Ensure valid JSON syntax"
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            few_shot_prompt,
            ("user", "Question:\n{question}")
        ])

    def _make_api_request(self, prompt: str, max_retries: int = 3) -> str:
        """Make direct API request to HuggingFace"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": 0.01,
                "top_p": 0.95,
                "repetition_penalty": 1.1,
                "return_full_text": False,
                "stop": ["Question:", "User:", "Human:", "\n\nQuestion"]
            },
            "options": {
                "wait_for_model": True,
                "use_cache": False
            }
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 503:
                    # Model is loading, wait and retry
                    time.sleep(10)
                    continue

                if response.status_code == 429:
                    # Rate limited, wait and retry
                    time.sleep(5)
                    continue

                response.raise_for_status()
                result = response.json()

                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "")
                elif isinstance(result, dict) and "generated_text" in result:
                    return result["generated_text"]
                else:
                    raise ValueError(f"Unexpected response format: {result}")

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"API request failed after {max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff

        return ""

    def clean_response(self, response: str) -> str:
        """Clean and extract JSON from model response"""
        response = response.strip()
        response = re.sub(r'^```json\s*', '', response)
        response = re.sub(r'\s*```$', '', response)
        response = re.sub(r'^(AI|Answer):\s*', '', response, flags=re.IGNORECASE)

        # Find JSON pattern
        json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)

        if matches:
            largest_json = max(matches, key=len)
            return largest_json.strip()

        return response.strip()

    def validate_and_parse_json(self, json_str: str) -> Dict[str, Any]:
        """Validate JSON and ensure it matches expected structure"""
        try:
            parsed = json.loads(json_str)
            required_fields = [
                "problem_name", "input_format", "output_format",
                "constraints", "logic_summary", "dsa_topics", "edge_cases"
            ]

            for field in required_fields:
                if field not in parsed:
                    parsed[field] = f"Missing {field}"

            list_fields = ["input_format", "constraints", "dsa_topics", "edge_cases"]
            for field in list_fields:
                if not isinstance(parsed.get(field), list):
                    parsed[field] = [str(parsed.get(field, ""))]

            if not isinstance(parsed.get("output_format"), dict):
                parsed["output_format"] = {"type": "unknown", "description": str(parsed.get("output_format", ""))}

            return parsed

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {str(e)}")

    def analyze(self, question: str, max_retries: int = 3) -> str:
        """Analyze problem and return JSON string"""
        for attempt in range(max_retries):
            try:
                # Format the prompt using the template
                formatted_prompt = self.prompt_template.format(question=question)

                # Convert ChatPromptTemplate to string for API
                if hasattr(formatted_prompt, 'to_string'):
                    prompt_str = formatted_prompt.to_string()
                else:
                    # Manual formatting for direct API
                    prompt_str = f"""You are a coding problem analyzer. You MUST respond with ONLY valid JSON in this exact format:

{{
  "problem_name": "string",
  "input_format": [{{"name": "string", "type": "string"}}],
  "output_format": {{"type": "string", "description": "string"}},
  "constraints": ["string"],
  "logic_summary": "string",
  "dsa_topics": ["string"],
  "edge_cases": ["string"]
}}

IMPORTANT RULES:
- Respond with ONLY the JSON object
- Do NOT add any text before or after the JSON
- Do NOT use markdown formatting
- Ensure all strings are properly quoted
- Ensure valid JSON syntax

Question:
{question}

Answer:"""

                response = self._make_api_request(prompt_str)
                cleaned_response = self.clean_response(response)
                parsed_result = self.validate_and_parse_json(cleaned_response)

                return json.dumps(parsed_result)

            except Exception as e:
                if attempt == max_retries - 1:
                    fallback = {
                        "problem_name": "Analysis Failed",
                        "input_format": [{"name": "unknown", "type": "unknown"}],
                        "output_format": {"type": "unknown", "description": "Failed to parse response"},
                        "constraints": ["Unable to determine from response"],
                        "logic_summary": f"Failed to parse after {max_retries} attempts. Error: {str(e)}",
                        "dsa_topics": ["Unknown"],
                        "edge_cases": ["Unable to determine"]
                    }
                    return json.dumps(fallback)
                time.sleep(1)
                continue

        return "{}"


# Agent 2: Solution Generator (OpenAI) - Unchanged
class SolutionGenerator:
    def __init__(self):
        self.memory = LightweightMemory(max_pairs=3)
        self.system_prompt = """You are a DSA expert. For each problem if you are given that in JSON format with different constraints then:
1. Write clean, efficient code
2. Add key comments only
3. Include time/space complexity
4. Handle edge cases
5. Follow exact I/O format

if you have been given just a plain text question then don't write code give its answer in plain text only.

Keep responses under 80 lines. Use specified language."""

    def _extract_problem_info(self, problem_json: str) -> tuple:
        try:
            data = json.loads(problem_json)
            name = data.get("problem_name", "Unknown")
            topics = data.get("dsa_topics", ["General"])
            topic = topics[0] if topics else "General"
            return name, topic
        except:
            return "Unknown", "General"

    def solve(self, problem_json: str) -> str:
        problem_name, topic = self._extract_problem_info(problem_json)
        messages = [{"role": "system", "content": self.system_prompt}]

        context = self.memory.get_compact_context()
        user_content = problem_json
        if context:
            user_content = f"Context: {context}\n\nProblem: {problem_json}"
        else:
            user_content = f"Problem: {problem_json}"

        messages.append({"role": "user", "content": user_content})

        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )

            response_dict = response if isinstance(response, dict) else response.model_dump()

            if isinstance(response_dict, dict) and "choices" in response_dict:
                content = response_dict["choices"][0]["message"]["content"]
            elif "output" in response_dict:
                content = response_dict["output"]
            elif hasattr(response, "choices"):
                content = response.choices[0].message.content
            else:
                raise ValueError(f"Unexpected OpenAI response format: {response_dict}")

            content = self._extract_code_from_text(content)
            self.memory.add_qa_pair(problem_name, topic, user_content, content)

            return content

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return error_msg

    def _extract_code_from_text(self, content: str) -> str:
        if "```" in content:
            lines = content.split('\n')
            in_code = False
            code_lines = []
            for line in lines:
                if line.startswith('```'):
                    in_code = not in_code
                    continue
                if in_code:
                    code_lines.append(line)
            content = '\n'.join(code_lines) if code_lines else content
        return content.strip()


# Create the chain
def create_dsa_chain() -> Runnable:
    """Creates the complete LCEL chain: Agent 1 → Agent 2"""
    analyzer = ProblemAnalyzer()
    solver = SolutionGenerator()

    def agent1_wrapper(question_input: Dict[str, str]) -> str:
        question = question_input.get("question", "")
        if not question:
            raise ValueError("Question cannot be empty")
        return analyzer.analyze(question)

    def agent2_wrapper(problem_json: str) -> str:
        return solver.solve(str(problem_json))

    chain = (
            RunnableLambda(agent1_wrapper) |
            RunnableLambda(agent2_wrapper)
    )

    return chain


# Streamlit App - Unchanged
def main():
    st.set_page_config(
        page_title="Dual Agent DSA Solver",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Code Crafter")
    st.markdown(
        "Craft solutions to any Data Structures and Algorithms questions with our State-of-the-Art DSA Problem Solver ")
    st.markdown("**Agent 1** (HuggingFace LLM) analyzes problems → **Agent 2** (OpenAI LLM) generates solutions")

    # Initialize the chain in session state
    if 'chain' not in st.session_state:
        st.session_state.chain = create_dsa_chain()
        st.session_state.solutions_history = []

    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        st.success("Agent 1: HuggingFace LLM")
        st.success("Agent 2: OpenAI LLM ")

        st.markdown("---")
        st.markdown("### 📊 Session Stats")
        st.metric("Problems Solved", len(st.session_state.solutions_history))

        if st.session_state.solutions_history:
            recent_topics = set()
            for sol in st.session_state.solutions_history[-3:]:
                if 'analysis' in sol:
                    try:
                        analysis = json.loads(sol['analysis'])
                        recent_topics.update(analysis.get('dsa_topics', []))
                    except:
                        pass

            if recent_topics:
                st.write("**Recent Topics:**")
                for topic in list(recent_topics)[:5]:
                    st.write(f"• {topic}")

        st.markdown("---")
        st.markdown("### ℹ️ How it works:")
        st.markdown("1. **Agent 1** analyzes your problem using HuggingFace Agent")
        st.markdown("2. **Agent 2** generates solution using OpenAI LLM")
        st.markdown("3. Both agents work together seamlessly via LangChain")

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Problem Input")

        # Sample problems
        sample_problems = {
            "Two Sum": """Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You can return the answer in any order.

Example:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]""",

            "Binary Search": """Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

Example:
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4""",

            "Valid Parentheses": """Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

Example:
Input: s = "()[]{}"
Output: true""",

            "Longest Substring": """Given a string s, find the length of the longest substring without repeating characters.

Example:
Input: s = "abcabcbb"
Output: 3"""
        }

        selected_sample = st.selectbox(
            "Choose a sample problem (optional):",
            ["Custom"] + list(sample_problems.keys())
        )

        if selected_sample != "Custom":
            problem_text = st.text_area(
                "Problem Description:",
                value=sample_problems[selected_sample],
                height=200
            )
        else:
            problem_text = st.text_area(
                "Problem Description:",
                placeholder="Paste your coding problem here...",
                height=200
            )

        solve_button = st.button("🚀 Solve Problem", type="primary", disabled=not problem_text)

    with col2:
        st.subheader("⚡ Quick Actions")

        if st.button("🗑️ Clear History"):
            st.session_state.solutions_history = []
            st.rerun()

        if st.button("🔄 Reset Chain"):
            st.session_state.chain = create_dsa_chain()
            st.success("Chain reset successfully!")

    # Process problem solving
    if solve_button and problem_text:
        with st.spinner("🔍 Agent 1: Analyzing problem..."):
            try:
                # Create progress placeholder
                progress_placeholder = st.empty()

                # Step 1: Analysis
                progress_placeholder.info("🔍 Agent 1: Analyzing problem structure...")
                time.sleep(1)

                # Step 2: Solution
                progress_placeholder.info("🚀 Agent 2: Generating solution...")

                # Invoke the chain
                question_input = {"question": problem_text}
                result = st.session_state.chain.invoke(question_input)

                progress_placeholder.success("✅ Solution completed!")

                # Store the result (we need to get the analysis separately for display)
                analyzer = ProblemAnalyzer()
                analysis_json = analyzer.analyze(problem_text)

                st.session_state.solutions_history.insert(0, {
                    'problem': problem_text[:100] + "..." if len(problem_text) > 100 else problem_text,
                    'analysis': analysis_json,
                    'solution': result,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Display results
    if st.session_state.solutions_history:
        st.markdown("---")
        st.subheader("🎯 Latest Solution")

        latest = st.session_state.solutions_history[0]

        # Parse analysis for display
        try:
            analysis = json.loads(latest['analysis'])

            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Analysis", "💻 Solution", "🏷️ Details", "📊 Breakdown"])

            with tab1:
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Problem Name:**")
                    st.write(analysis.get('problem_name', 'Unknown'))

                    st.write("**DSA Topics:**")
                    topics = analysis.get('dsa_topics', [])
                    for topic in topics:
                        st.badge(topic)

                with col2:
                    st.write("**Logic Summary:**")
                    st.write(analysis.get('logic_summary', 'Not available'))

            with tab2:
                st.code(latest['solution'], language="python")

                # Download button
                st.download_button(
                    label="📥 Download Solution",
                    data=latest['solution'],
                    file_name=f"{analysis.get('problem_name', 'solution').replace(' ', '_').lower()}.py",
                    mime="text/plain"
                )

            with tab3:
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Input Format:**")
                    input_format = analysis.get('input_format', [])
                    for inp in input_format:
                        st.write(f"• **{inp.get('name', 'unknown')}**: {inp.get('type', 'unknown')}")

                    st.write("**Constraints:**")
                    constraints = analysis.get('constraints', [])
                    for constraint in constraints:
                        st.write(f"• {constraint}")

                with col2:
                    st.write("**Output Format:**")
                    output = analysis.get('output_format', {})
                    st.write(f"• **Type**: {output.get('type', 'unknown')}")
                    st.write(f"• **Description**: {output.get('description', 'Not available')}")

                    st.write("**Edge Cases:**")
                    edge_cases = analysis.get('edge_cases', [])
                    for case in edge_cases:
                        st.write(f"• {case}")

            with tab4:
                # Agent workflow visualization
                st.write("**Agent Workflow:**")

                workflow_data = {
                    "Agent 1 (HuggingFace)": "Problem Analysis",
                    "→": "JSON Structure",
                    "Agent 2 (OpenAI)": "Solution Generation"
                }

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info("🤖 **Agent 1**\nHuggingFace Direct API\nProblem Analysis")
                with col2:
                    st.info("📊 **Transfer**\nStructured JSON\nData Format")
                with col3:
                    st.info("🚀 **Agent 2**\nOpenAI GPT-3.5\nSolution Code")

        except json.JSONDecodeError:
            st.error("Failed to parse analysis data")
            st.code(latest['solution'], language="python")

        # Show previous solutions
        if len(st.session_state.solutions_history) > 1:
            st.markdown("---")
            st.subheader("📚 Previous Solutions")

            for i, sol in enumerate(st.session_state.solutions_history[1:6], 1):  # Show last 5
                try:
                    sol_analysis = json.loads(sol['analysis'])
                    problem_name = sol_analysis.get('problem_name', 'Unknown')
                except:
                    problem_name = f"Solution {i}"

                with st.expander(f"**{problem_name}** ({sol['timestamp']})"):
                    st.code(sol['solution'], language="python")


if __name__ == "__main__":
    main()
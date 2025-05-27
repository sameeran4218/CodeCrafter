# CodeCrafter

app -> https://codecrafter4218.streamlit.app/

demo -> https://youtu.be/n8twMXZUVkY

# 🤖 Code Crafter: Dual-Agent DSA Problem Solver

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code Crafter is an intelligent system that combines two AI agents to analyze and solve Data Structures and Algorithms (DSA) problems. Agent 1 (HuggingFace) performs problem analysis, while Agent 2 (OpenAI) generates optimized solutions.

![App Screenshot](docs/screenshot.png)

## 🌟 Features

- **Dual-Agent Architecture**: Two specialized LLMs working in tandem
- **Structured Problem Analysis**: Extracts input/output formats, constraints, and edge cases
- **Optimized Solutions**: Generates clean, efficient code with complexity analysis
- **Interactive UI**: Streamlit-based interface with sample problems and history
- **Lightweight Memory**: Maintains session context without heavy storage
- **Multi-language Support**: Primarily Python but adaptable to other languages

## 🛠️ Tech Stack

### Core Components
| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Agent 1 (Analysis) | HuggingFace (Zephyr-7B) |
| Agent 2 (Solution) | OpenAI (GPT-3.5-turbo) |
| Memory System | Custom lightweight implementation |
| Orchestration | LangChain (LCEL) |

### Supporting Libraries
- Pydantic (for data validation)
- Requests (for API calls)
- Python-dotenv (for environment variables)

## 🏗️ System Architecture

```mermaid
graph TD
    A[User Input] --> B[Streamlit UI]
    B --> C[Agent 1: Problem Analyzer]
    C --> D[Structured JSON Analysis]
    D --> E[Agent 2: Solution Generator]
    E --> F[Optimized Solution]
    F --> G[User Output]
    C --> H[Lightweight Memory]
    E --> H
    H --> C
    H --> E

# 🚀 LangGraph Code Assistant

**🧠 AI-powered Python Code Generation with Retrieval-Augmented Generation (RAG) and Iterative Improvement**

---

## ✨ Overview

LangGraph Code Assistant is a Streamlit-based web app that leverages cutting-edge language models and vector search to help you generate clean, well-commented, and executable Python code from natural language prompts. It combines:

* 📚 **Context-aware code generation** using your uploaded docs or pasted context
* 🔄 **Iterative code refinement** with automated error detection and fixes
* 🛠️ **Minimal sandboxed code execution** for safe testing
* 🎛️ **Interactive UI** for API keys, model config, and detailed logs

---

## 🎯 Features

* 📂 Upload multiple documentation or code files (`.txt`, `.md`, `.py`) to build your knowledge base
* ✍️ Paste relevant context directly for prompt enrichment
* 💬 Ask coding questions or describe tasks naturally
* 🔍 Retrieval-Augmented Generation (RAG) grounds responses on your specific project context
* 🧩 Automated detection and insertion of missing Python imports
* 🤖 Iterative code generation with runtime feedback loops
* ⏱️ Execute code in a temporary isolated environment with timeout limits
* 👀 View generated code, execution logs, and debug info side-by-side
* 💾 Download generated Python scripts with a single click

---

## 📸 Demo Screenshot

*(Add a screenshot here to showcase the sleek UI)*

---

## 🚀 Getting Started

### 🛠️ Prerequisites

* Python 3.8+
* OpenAI API key (required)
* (Optional) Anthropic API key for additional LLM support

### 📦 Installation

```bash
git clone https://github.com/yPRAJINRAJ-RANAWAT/Langgraph-Code-Generator.git
cd langgraph-code-assistant
pip install -r requirements.txt
```

### ▶️ Run the App

```bash
streamlit run streamlit_langgraph_code_assistant.py
```

Open your browser at [http://localhost:8501](http://localhost:8501)

---

## 🧩 Usage

1. Enter your OpenAI API key (and Anthropic key if available) in the sidebar
2. Upload your project documentation or paste context for background info
3. Describe your coding task or question in the text area
4. Click **Generate Code** and watch the magic happen!
5. Review generated code, execution logs, and debug info
6. Download the Python script if satisfied

---

## 🔍 How It Works

1. **Context Processing**

   * Combine uploaded files and manual context, split into chunks
   * Generate vector embeddings for similarity search

2. **Retrieval-Augmented Generation (RAG)**

   * Retrieve top relevant context chunks for your query
   * Use LLM with context to generate Python code

3. **Code Analysis & Execution**

   * Detect & insert missing imports automatically
   * Execute generated code safely with timeout and capture output
   * On errors, provide feedback to LLM for fixes (iterative refinement)

4. **Result Presentation**

   * Display final code, logs, and debug data in UI
   * Provide download link for generated script

---

## 🔐 Security Notice

⚠️ This app runs generated code in a **minimal sandbox** (isolated subprocess with timeout).
**Do NOT run untrusted code on production servers without stronger sandboxing measures** (Docker, firejail, gVisor, etc.)

---

## 🛠️ Technologies Used

* [Streamlit](https://streamlit.io) — interactive UI
* [LangChain](https://langchain.com) — chaining LLM calls & embeddings
* OpenAI GPT models (`gpt-4o-mini`, `gpt-4`) — code generation
* FAISS — fast similarity search over context chunks
* Python stdlib — subprocess management, regex, and more

---

## 📬 Contact

Created by **Prajin Ranawat**
[Linkedin] (https://www.linkedin.com/in/prajinraj-ranawat-8257952a2/)


Would you like me to generate that `requirements.txt` for your project next?

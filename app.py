# streamlit_langgraph_code_assistant.py
"""
Streamlit app for LangGraph-powered Code Assistant
Features:
- Upload project docs or paste context
- Ask a coding question / task
- Run LangGraph pipeline: RAG -> Generate Code -> Import Check -> Execute Test
- Show logs, iterative corrections, and final code

NOTE: This is a single-file starter app. You must have API keys set as environment variables
(or enter them in the sidebar): OPENAI_API_KEY (and/or ANTHROPIC_API_KEY)

Dependencies: See requirements.txt

Security: The "run code" step uses a very small sandbox approach using subprocess with timeouts.
Do NOT run untrusted code on production servers without stronger sandboxing (docker, firejail, gVisor).
"""

import streamlit as st
import os
import tempfile
import subprocess
import textwrap
import uuid
import time
import re
from typing import Optional, Dict, Any, List
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set page config
st.set_page_config(page_title="LangGraph Code Assistant", layout="wide")

# --- Helper utilities ---

def set_api_keys(openai_key: Optional[str], anthropic_key: Optional[str]):
    """Set API keys in environment variables"""
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key


def safe_execute_python(code: str, timeout: int = 10) -> Dict[str, Any]:
    """Run python code in a temporary file using subprocess, capture stdout/stderr.
    This is a minimal sandbox: enforced timeout and process isolation only. 
    Use stronger sandboxing for production."""
    tmp_dir = tempfile.mkdtemp(prefix="code_exec_")
    filename = os.path.join(tmp_dir, f"script_{uuid.uuid4().hex}.py")
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)

        start = time.time()
        proc = subprocess.run(
            ["python", filename], 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=tmp_dir
        )
        elapsed = time.time() - start
        
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed": elapsed,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1, 
            "stdout": "", 
            "stderr": f"Timeout after {timeout}s", 
            "elapsed": timeout
        }
    except Exception as e:
        return {
            "returncode": -2, 
            "stdout": "", 
            "stderr": str(e), 
            "elapsed": 0
        }
    finally:
        # Clean up temp file
        try:
            os.remove(filename)
            os.rmdir(tmp_dir)
        except:
            pass


def extract_code_from_text(text: str) -> Optional[str]:
    """Extract Python code from text, looking for code blocks first"""
    # Look for triple backtick code blocks
    code_block_pattern = r"```(?:python)?\n?(.*?)```"
    matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return matches[0].strip()
    
    # Fallback: if text looks like code (contains common Python keywords)
    python_indicators = ["def ", "import ", "from ", "print(", "class ", "if __name__"]
    if any(indicator in text for indicator in python_indicators):
        return text.strip()
    
    return None


def find_missing_imports(code: str) -> List[str]:
    """Detect potentially missing imports using simple heuristics"""
    # Common library patterns
    import_patterns = {
        'numpy': [r'\bnp\.', r'numpy\.'],
        'pandas': [r'\bpd\.', r'pandas\.'],
        'matplotlib': [r'plt\.', r'matplotlib\.'],
        'requests': [r'requests\.'],
        'json': [r'json\.'],
        'os': [r'\bos\.'],
        'sys': [r'\bsys\.'],
        'time': [r'\btime\.'],
        'datetime': [r'\bdatetime\.', r'\bdt\.'],
        'random': [r'\brandom\.'],
        'math': [r'\bmath\.'],
        're': [r'\bre\.']
    }
    
    # Get existing imports
    existing_imports = set()
    import_lines = re.findall(r'^(?:from\s+(\w+)|import\s+(\w+))', code, re.MULTILINE)
    for match in import_lines:
        existing_imports.update(filter(None, match))
    
    # Check for missing imports
    missing = []
    for lib, patterns in import_patterns.items():
        if lib not in existing_imports:
            for pattern in patterns:
                if re.search(pattern, code):
                    missing.append(lib)
                    break
    
    return missing


def build_rag_pipeline(embedding_model_name: str = "text-embedding-3-small", 
                      llm_model_name: str = "gpt-4o-mini"):
    """Build a RAG pipeline for code generation with iterative improvement"""
    
    def pipeline(context_text: str, user_question: str, max_iters: int = 3) -> Dict[str, Any]:
        """Run RAG -> generate -> check imports -> execute iteratively"""
        logs = []
        
        # Initialize components
        try:
            embeddings = OpenAIEmbeddings(model=embedding_model_name)
            llm = ChatOpenAI(model=llm_model_name, temperature=0)
            logs.append(f"✓ Initialized LLM ({llm_model_name}) and embeddings")
        except Exception as e:
            logs.append(f"✗ Failed to initialize components: {e}")
            return {"logs": logs, "final_code": None, "last_exec": None}
        
        # Create vectorstore from context
        vectorstore = None
        if context_text.strip():
            try:
                # Split context into chunks for better retrieval
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                chunks = text_splitter.split_text(context_text)
                vectorstore = FAISS.from_texts(chunks, embeddings)
                logs.append(f"✓ Created vectorstore with {len(chunks)} chunks")
            except Exception as e:
                logs.append(f"⚠ Vectorstore creation failed: {e}")
        
        # Retrieve relevant context
        retrieved_context = context_text
        if vectorstore:
            try:
                docs = vectorstore.similarity_search(user_question, k=3)
                retrieved_context = "\n\n".join([doc.page_content for doc in docs])
                logs.append(f"✓ Retrieved {len(docs)} relevant chunks")
            except Exception as e:
                logs.append(f"⚠ Retrieval failed: {e}")
        
        # Code generation prompt
        code_prompt = PromptTemplate(
            input_variables=["context", "question", "feedback"],
            template="""You are an expert Python developer. Use the provided context to answer the coding question.

Context:
{context}

Question: {question}

{feedback}

Important instructions:
1. Provide ONLY executable Python code
2. Include all necessary imports at the top
3. Write clean, well-commented code
4. Handle potential errors gracefully
5. If creating functions, include a simple test/example at the bottom

Return your code wrapped in triple backticks with 'python' language specification."""
        )
        
        chain = LLMChain(llm=llm, prompt=code_prompt)
        
        final_code = None
        last_exec_result = None
        
        # Iterative improvement loop
        for iteration in range(max_iters):
            logs.append(f"🔄 Iteration {iteration + 1}/{max_iters}")
            
            # Prepare feedback for subsequent iterations
            feedback = ""
            if iteration > 0 and last_exec_result:
                if last_exec_result["returncode"] != 0:
                    feedback = f"""
Previous attempt failed with error:
{last_exec_result['stderr']}

Please fix the error and provide corrected code."""
                else:
                    break  # Success, exit early
            
            # Generate code
            try:
                response = chain.run({
                    "context": retrieved_context,
                    "question": user_question,
                    "feedback": feedback
                })
                
                code = extract_code_from_text(response)
                if not code:
                    logs.append("⚠ No code block found in LLM response, using full response")
                    code = response.strip()
                
                logs.append(f"✓ Generated code ({len(code)} characters)")
                
            except Exception as e:
                logs.append(f"✗ Code generation failed: {e}")
                continue
            
            # Check for missing imports
            missing_imports = find_missing_imports(code)
            if missing_imports:
                logs.append(f"⚠ Potentially missing imports: {', '.join(missing_imports)}")
                # Add imports if they seem to be missing
                import_lines = '\n'.join([f"import {imp}" for imp in missing_imports])
                code = import_lines + '\n\n' + code
            
            # Test execution
            logs.append("🧪 Testing code execution...")
            exec_result = safe_execute_python(code, timeout=10)
            last_exec_result = exec_result
            
            if exec_result["returncode"] == 0:
                logs.append(f"✅ Code executed successfully in {exec_result['elapsed']:.2f}s")
                if exec_result["stdout"]:
                    logs.append(f"📤 Output: {exec_result['stdout'][:200]}...")
                final_code = code
                break
            else:
                logs.append(f"❌ Execution failed (code {exec_result['returncode']})")
                logs.append(f"🔍 Error: {exec_result['stderr'][:300]}...")
        
        if not final_code:
            logs.append("⚠ No successful code generated after all iterations")
        
        return {
            "logs": logs, 
            "final_code": final_code, 
            "last_exec": last_exec_result
        }
    
    return pipeline


# --- Streamlit UI ---

st.title("🚀 LangGraph Code Assistant")
st.markdown("*AI-powered code generation with RAG and iterative improvement*")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Keys
    openai_key = st.text_input("OpenAI API Key", type="password", 
                              help="Required for LLM and embeddings")
    anthropic_key = st.text_input("Anthropic API Key (optional)", type="password")
    
    # Model selection
    model_choice = st.selectbox(
        "LLM Model", 
        ["gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0,
        help="Choose the OpenAI model for code generation"
    )
    
    embedding_model = st.selectbox(
        "Embedding Model",
        ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
        index=0
    )
    
    # Set API keys button
    if st.button("🔑 Set API Keys", type="primary"):
        if not openai_key:
            st.error("OpenAI API key is required!")
        else:
            set_api_keys(openai_key, anthropic_key)
            st.success("✅ API keys configured!")
    
    st.divider()
    st.markdown("### 🔧 Pipeline Settings")
    max_iters = st.slider("Max iterations", 1, 5, 3, 
                         help="Maximum attempts for code generation and fixing")

# Main interface
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📚 Context & Documentation")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload documentation files", 
        accept_multiple_files=True,
        type=['txt', 'md', 'py'],
        help="Upload relevant documentation or code files"
    )
    
    # Manual context input
    manual_context = st.text_area(
        "Or paste context here", 
        height=200,
        placeholder="Paste relevant code, documentation, or context here..."
    )
    
    # Process uploaded files
    context_text = manual_context
    if uploaded_files:
        file_contents = []
        for uploaded_file in uploaded_files:
            try:
                content = uploaded_file.read().decode("utf-8")
                file_contents.append(f"# File: {uploaded_file.name}\n{content}")
            except Exception as e:
                st.warning(f"Could not read {uploaded_file.name}: {e}")
        
        if file_contents:
            uploaded_context = "\n\n" + "\n\n".join(file_contents)
            context_text = (context_text + uploaded_context).strip()
    
    st.divider()
    
    # Coding task input
    st.subheader("💬 Coding Task")
    question = st.text_area(
        "Describe your coding task or question",
        height=150,
        placeholder="e.g., Create a function to parse CSV files and return statistics..."
    )
    
    # Generate button
    generate_btn = st.button("🚀 Generate Code", type="primary", use_container_width=True)

with col2:
    st.subheader("📋 Results")
    
    if generate_btn:
        if not question.strip():
            st.error("❌ Please provide a coding task or question!")
        elif not openai_key:
            st.error("❌ Please set your OpenAI API key in the sidebar!")
        else:
            # Create pipeline and run
            with st.spinner("🔄 Running RAG pipeline..."):
                pipeline = build_rag_pipeline(embedding_model, model_choice)
                result = pipeline(context_text, question, max_iters)
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["📝 Generated Code", "📊 Execution Logs", "🔍 Debug Info"])
            
            with tab1:
                final_code = result.get("final_code")
                if final_code:
                    st.success("✅ Code generated successfully!")
                    st.code(final_code, language="python")
                    
                    # Download button
                    st.download_button(
                        label="💾 Download Code",
                        data=final_code,
                        file_name="generated_code.py",
                        mime="text/plain"
                    )
                else:
                    st.warning("⚠️ No successful code generated. Check logs for details.")
            
            with tab2:
                logs = result.get("logs", [])
                if logs:
                    for log in logs:
                        st.text(log)
                else:
                    st.info("No logs available.")
            
            with tab3:
                last_exec = result.get("last_exec")
                if last_exec:
                    st.json(last_exec)
                else:
                    st.info("No execution information available.")
    
    else:
        st.info("👆 Configure your settings and click 'Generate Code' to start!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>
    ⚠️ <strong>Security Notice:</strong> This app executes generated code in a minimal sandbox. 
    Do not run in production environments without proper security measures.
    </small>
</div>
""", unsafe_allow_html=True)
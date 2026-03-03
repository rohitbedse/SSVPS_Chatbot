# ------------------------------------------------------------------------------------
# 0️⃣ Install requirements:
# pip install streamlit langchain langchain-google-genai pypdf faiss-cpu python-dotenv
# 
# Place this script in the same folder as: ssvps_college_dataset_unicode.pdf
# ------------------------------------------------------------------------------------

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# CONFIGURATION
PDF_FILE = "ssvps_college_dataset_unicode.pdf"  # Your specific PDF
INDEX_DIR = "ssvps_gemini_index"                # Where to save/load index

# Page config
st.set_page_config(
    page_title="SSVPS College RAG",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { 
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        font-weight: 800;
    }
    .stChatMessage { 
        padding: 1rem; 
        border-radius: 1rem; 
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .source-box { 
        background: #f8f9fa; 
        padding: 0.75rem; 
        border-radius: 0.5rem; 
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #555;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-ready { background: #d4edda; color: #155724; }
    .status-loading { background: #fff3cd; color: #856404; }
    .status-error { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def initialize_rag_system(api_key):
    """Initialize or load existing RAG system"""
    
    # Setup embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )
    
    # Check if index exists
    if os.path.exists(INDEX_DIR):
        st.info("📂 Loading existing knowledge base...")
        vectorstore = FAISS.load_local(
            INDEX_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vectorstore
    
    # Create new index from PDF
    if not os.path.exists(PDF_FILE):
        raise FileNotFoundError(f"PDF not found: {PDF_FILE}")
    
    st.info("🔨 Building knowledge base from PDF... (one-time setup)")
    
    # Load and process PDF
    loader = PyPDFLoader(PDF_FILE)
    pages = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(pages)
    
    # Create vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    
    st.success(f"✅ Indexed {len(pages)} pages into {len(chunks)} chunks!")
    return vectorstore

def get_qa_chain(vectorstore, api_key):
    """Create QA chain"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.2
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Load environment variables
load_dotenv()
env_api_key = os.getenv("GOOGLE_API_KEY")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key handling
    api_key = st.text_input(
        "Google API Key",
        type="password",
        value=env_api_key if env_api_key else "",
        help="Leave empty if set in .env file"
    )
    
    final_api_key = api_key or env_api_key
    
    st.divider()
    
    # System status
    st.subheader("📊 System Status")
    
    if not final_api_key:
        st.markdown('<span class="status-badge status-error">🔴 API Key Missing</span>', unsafe_allow_html=True)
    elif st.session_state.qa_chain:
        st.markdown('<span class="status-badge status-ready">🟢 Ready</span>', unsafe_allow_html=True)
        st.caption(f"📄 Using: `{PDF_FILE}`")
        if os.path.exists(INDEX_DIR):
            st.caption(f"💾 Index cached locally")
    else:
        st.markdown('<span class="status-badge status-loading">🟡 Initializing...</span>', unsafe_allow_html=True)
    
    st.divider()
    
    # Management options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Rebuild Index", use_container_width=True):
            if os.path.exists(INDEX_DIR):
                import shutil
                shutil.rmtree(INDEX_DIR)
                st.session_state.vectorstore = None
                st.session_state.qa_chain = None
                st.success("Index cleared! Reload to rebuild.")
                st.rerun()
    
    st.divider()
    st.caption("💡 Tip: Index auto-saves after first run")

# Main UI
st.markdown("<h1 class='main-header'>🎓 SSVPS College Knowledge Assistant</h1>", unsafe_allow_html=True)
st.caption("Powered by Gemini 2.0 • RAG-Enabled Document Q&A")

# Initialize RAG system automatically
if final_api_key and not st.session_state.qa_chain:
    try:
        with st.spinner("⚡ Initializing knowledge base..."):
            vectorstore = initialize_rag_system(final_api_key)
            st.session_state.vectorstore = vectorstore
            st.session_state.qa_chain = get_qa_chain(vectorstore, final_api_key)
            st.rerun()
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        st.stop()

# Show chat interface only when ready
if st.session_state.qa_chain:
    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍🎓" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📎 Sources", expanded=False):
                    for i, src in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <b>Source {i}</b> • {src['meta']}<br>
                            <i>{src['text'][:250]}...</i>
                        </div>
                        """, unsafe_allow_html=True)

    # Input
    if prompt := st.chat_input("Ask about SSVPS College..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🧠 Retrieving information..."):
                try:
                    result = st.session_state.qa_chain({"query": prompt})
                    answer = result["result"]
                    sources = result["source_documents"]
                    
                    st.markdown(answer)
                    
                    # Process sources
                    source_data = []
                    with st.expander("📎 Sources", expanded=False):
                        for i, doc in enumerate(sources, 1):
                            src = {
                                "meta": doc.metadata.get("source", "PDF"),
                                "text": doc.page_content
                            }
                            source_data.append(src)
                            st.markdown(f"""
                            <div class="source-box">
                                <b>Source {i}</b> • {doc.metadata.get("source", "PDF")}<br>
                                <i>{doc.page_content[:250]}...</i>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": source_data
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

else:
    # Not ready state
    st.info("""
    ### 🚀 Getting Started
    
    1. **Set your Google API Key**:
       - Enter in sidebar, OR
       - Create `.env` file with: `GOOGLE_API_KEY=your_key_here`
    
    2. **Ensure PDF exists**: `{PDF_FILE}` in same folder
    
    3. **First run**: System will auto-index the PDF (saved to `{INDEX_DIR}/`)
    
    4. **Subsequent runs**: Loads instantly from cache!
    """)
    
    if not os.path.exists(PDF_FILE):
        st.error(f"⚠️ PDF file not found: `{PDF_FILE}`")
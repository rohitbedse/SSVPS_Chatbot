# ---------------------------------------------------------------------------
# SSVPS College Knowledge Assistant – Production-Ready Streamlit Chatbot
# ---------------------------------------------------------------------------

import logging
import os
import shutil

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PDF_FILE = "ssvps_college_dataset_unicode.pdf"
INDEX_DIR = "ssvps_gemini_index"
MAX_INPUT_LENGTH = 1000

SUGGESTED_QUESTIONS = [
    "What courses are offered at SSVPS College?",
    "What is the fee structure for B.Tech?",
    "Tell me about the placement record",
    "What are the eligibility criteria for admission?",
    "Describe the college library facilities",
    "What sports facilities are available?",
]

SYSTEM_PROMPT = """You are the official AI assistant for SSVPS's Bapusaheb Shivajirao \
Deore College of Engineering, Dhule, Maharashtra. You provide accurate, helpful, and \
friendly answers about the college based **only** on the retrieved context below.

Guidelines:
- Answer in clear, well-structured sentences. Use bullet points or numbered lists when \
listing multiple items (courses, departments, fees, etc.).
- If the context does not contain enough information to fully answer the question, say \
so honestly and suggest the user contact the college office.
- When discussing fees, intake numbers, or eligibility, always mention the relevant \
academic year if available in the context.
- Be polite and professional; address the user respectfully.
- Do NOT fabricate information that is not present in the context.

Context:
{context}"""

HUMAN_PROMPT = "{question}"

# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SSVPS College Assistant",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
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
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
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
    .suggestion-btn button {
        width: 100%;
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
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def build_vectorstore(api_key: str):
    """Build or load the FAISS vectorstore from the PDF dataset."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
    )

    if os.path.exists(INDEX_DIR):
        logger.info("Loading existing FAISS index from %s", INDEX_DIR)
        return FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    if not os.path.exists(PDF_FILE):
        raise FileNotFoundError(f"PDF not found: {PDF_FILE}")

    logger.info("Building FAISS index from %s …", PDF_FILE)
    loader = PyPDFLoader(PDF_FILE)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    logger.info("Indexed %d pages → %d chunks", len(pages), len(chunks))
    return vectorstore


def get_qa_chain(vectorstore, api_key: str):
    """Create a conversational retrieval chain with memory."""
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10},
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(HUMAN_PROMPT),
        ]
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=6,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
load_dotenv()
env_api_key = os.getenv("GOOGLE_API_KEY")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Google API Key",
        type="password",
        value=env_api_key or "",
        help="Enter your Gemini API key or set GOOGLE_API_KEY in a .env file",
    )
    final_api_key = api_key or env_api_key

    st.divider()
    st.subheader("📊 System Status")

    if not final_api_key:
        st.markdown(
            '<span class="status-badge status-error">🔴 API Key Missing</span>',
            unsafe_allow_html=True,
        )
    elif st.session_state.qa_chain:
        st.markdown(
            '<span class="status-badge status-ready">🟢 Ready</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"📄 Dataset: `{PDF_FILE}`")
        if os.path.exists(INDEX_DIR):
            st.caption("💾 Index cached locally")
    else:
        st.markdown(
            '<span class="status-badge status-loading">🟡 Initializing…</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.qa_chain = None  # reset memory
            st.rerun()
    with col2:
        if st.button("🔄 Rebuild Index", use_container_width=True):
            if os.path.exists(INDEX_DIR):
                shutil.rmtree(INDEX_DIR)
            st.session_state.qa_chain = None
            build_vectorstore.clear()
            st.rerun()

    st.divider()
    st.caption("💡 Index auto-saves after the first run and loads instantly on restart.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 class='main-header'>🎓 SSVPS College Knowledge Assistant</h1>",
    unsafe_allow_html=True,
)
st.caption("Powered by Gemini 2.0 Flash • RAG-Enabled Document Q&A • Conversation-Aware")

# ---------------------------------------------------------------------------
# Initialize RAG system
# ---------------------------------------------------------------------------
if final_api_key and not st.session_state.qa_chain:
    try:
        with st.spinner("⚡ Initializing knowledge base…"):
            vectorstore = build_vectorstore(final_api_key)
            st.session_state.qa_chain = get_qa_chain(vectorstore, final_api_key)
            st.rerun()
    except FileNotFoundError as exc:
        st.error(f"❌ {exc}")
        st.stop()
    except Exception as exc:
        logger.exception("Initialization failed")
        st.error(f"❌ Initialization failed: {exc}")
        st.stop()


# ---------------------------------------------------------------------------
# Helper: handle a user query
# ---------------------------------------------------------------------------
def _handle_query(user_query: str):
    """Send *user_query* to the QA chain and render the response."""
    if len(user_query) > MAX_INPUT_LENGTH:
        st.warning(
            f"⚠️ Your question is too long ({len(user_query)} chars). "
            f"Please keep it under {MAX_INPUT_LENGTH} characters."
        )
        return

    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🧠 Retrieving information…"):
            try:
                result = st.session_state.qa_chain({"question": user_query})
                answer = result["answer"]
                sources = result.get("source_documents", [])
            except Exception as exc:
                logger.exception("Query failed")
                st.error(f"Something went wrong: {exc}")
                return

            st.markdown(answer)

            if sources:
                source_data = []
                with st.expander("📎 Sources", expanded=False):
                    for i, doc in enumerate(sources, 1):
                        page_num = doc.metadata.get("page", "–")
                        preview = doc.page_content[:300].replace("\n", " ")
                        src = {
                            "meta": f"Page {page_num}",
                            "text": doc.page_content,
                        }
                        source_data.append(src)
                        st.markdown(
                            f'<div class="source-box">'
                            f"<b>Source {i}</b> • Page {page_num}<br>"
                            f"<i>{preview}…</i></div>",
                            unsafe_allow_html=True,
                        )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": source_data}
                )
            else:
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )


# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------
if st.session_state.qa_chain:
    # Render history
    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📎 Sources", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        preview = src["text"][:300].replace("\n", " ")
                        st.markdown(
                            f'<div class="source-box">'
                            f"<b>Source {i}</b> • {src['meta']}<br>"
                            f"<i>{preview}…</i></div>",
                            unsafe_allow_html=True,
                        )

    # Suggested questions (shown only when chat is empty)
    if not st.session_state.messages:
        st.markdown("#### 💡 Try asking:")
        cols = st.columns(2)
        for idx, question in enumerate(SUGGESTED_QUESTIONS):
            with cols[idx % 2]:
                if st.button(question, key=f"sq_{idx}", use_container_width=True):
                    _handle_query(question)
                    st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask anything about SSVPS College…"):
        _handle_query(prompt)
        st.rerun()

else:
    # Not-ready state
    st.info(
        f"""
### 🚀 Getting Started

1. **Set your Google API Key** — enter it in the sidebar or create a `.env` file:
   ```
   GOOGLE_API_KEY=your_key_here
   ```
2. **Ensure the PDF dataset exists** — `{PDF_FILE}` must be in the project folder.
3. **First run** — the system will automatically index the PDF (saved to `{INDEX_DIR}/`).
4. **Subsequent runs** — the index loads instantly from cache!
"""
    )
    if not os.path.exists(PDF_FILE):
        st.error(f"⚠️ PDF file not found: `{PDF_FILE}`")
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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load API key from .env
# ---------------------------------------------------------------------------

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PDF_FILE = "ssvps_college_dataset_cleaned.pdf"
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

SMALL_TALK = [
    "hi",
    "hello",
    "hey",
    "good morning",
    "good evening",
    "thanks",
    "thank you",
    "ok",
    "okay",
]

SYSTEM_PROMPT = """
You are the official AI assistant for SSVPS's Bapusaheb Shivajirao Deore 
College of Engineering, Dhule, Maharashtra.

Answer questions ONLY using the retrieved context.

Rules:
- If the answer is not in the context say:
  "I could not find this information in the college dataset."
- Do NOT invent information.
- Use bullet points when helpful.

Context:
{context}
"""

HUMAN_PROMPT = "{question}"

# ---------------------------------------------------------------------------
# Streamlit Page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SSVPS College Assistant",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Embedding Loader
# ---------------------------------------------------------------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ---------------------------------------------------------------------------
# Vectorstore
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def build_vectorstore():

    embeddings = load_embeddings()

    if os.path.exists(INDEX_DIR):

        return FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    loader = PyPDFLoader(PDF_FILE)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(pages)

    for c in chunks:
        c.metadata["source"] = PDF_FILE

    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(INDEX_DIR)

    return vectorstore

# ---------------------------------------------------------------------------
# QA Chain
# ---------------------------------------------------------------------------

def get_qa_chain(vectorstore):

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20},
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=API_KEY,
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
# Small talk detector
# ---------------------------------------------------------------------------

def is_small_talk(query):

    q = query.lower().strip()

    return q in SMALL_TALK

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:

    st.header("⚙️ System")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Rebuild Index"):

        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)

        build_vectorstore.clear()

        st.session_state.qa_chain = None

        st.rerun()

    st.divider()

    st.caption("Dataset")
    st.code(PDF_FILE)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🎓 SSVPS College Knowledge Assistant")

st.caption("Gemini 2.5 Flash • RAG • Streamlit")

# ---------------------------------------------------------------------------
# Initialize RAG
# ---------------------------------------------------------------------------

if not st.session_state.qa_chain:

    with st.spinner("Loading knowledge base..."):

        vs = build_vectorstore()

        st.session_state.qa_chain = get_qa_chain(vs)

# ---------------------------------------------------------------------------
# Query Handler
# ---------------------------------------------------------------------------

def handle_query(user_query):

    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):

        if is_small_talk(user_query):

            answer = "Hello 👋 I'm the SSVPS College Assistant. How can I help you today?"

            st.markdown(answer)

        else:

            with st.spinner("Searching knowledge base..."):

                result = st.session_state.qa_chain(
                    {"question": user_query}
                )

                answer = result["answer"]

                sources = result.get("source_documents", [])

                st.markdown(answer)

                if sources:

                    with st.expander("Sources"):

                        for i, doc in enumerate(sources, 1):

                            page = doc.metadata.get("page", "-")

                            preview = doc.page_content[:250]

                            st.markdown(
                                f"**Source {i} • Page {page}**\n{preview}..."
                            )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

# ---------------------------------------------------------------------------
# Chat History
# ---------------------------------------------------------------------------

for m in st.session_state.messages:

    with st.chat_message(m["role"]):

        st.markdown(m["content"])

# ---------------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------------

if not st.session_state.messages:

    st.subheader("Try asking")

    cols = st.columns(2)

    for i, q in enumerate(SUGGESTED_QUESTIONS):

        with cols[i % 2]:

            if st.button(q):

                handle_query(q)

                st.rerun()

# ---------------------------------------------------------------------------
# Chat Input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask about SSVPS College"):

    handle_query(prompt)

    st.rerun()
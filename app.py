# ---------------------------------------------------------------------------
# SSVPS College Hybrid RAG Assistant (Semantic + Keyword)
# ---------------------------------------------------------------------------

import logging
import os
import shutil

import streamlit as st
from dotenv import load_dotenv

from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

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
# ENV
# ---------------------------------------------------------------------------

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PDF_FILE = "ssvps_college_dataset_cleaned.pdf"
INDEX_DIR = "faiss_index"

SYSTEM_PROMPT = """
You are the official AI assistant for SSVPS College.

STRICT RULES:
- Answer ONLY from context
- If not found say:
  "I could not find this information in the college dataset."
- Do NOT guess
- Keep answers clean and structured

Context:
{context}
"""

HUMAN_PROMPT = "{question}"

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ---------------------------------------------------------------------------
# Build Retrievers (FAISS + BM25)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def build_retrievers():

    embeddings = load_embeddings()

    loader = PyPDFLoader(PDF_FILE)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
    )

    docs = splitter.split_documents(pages)

    # ---------------- FAISS ----------------
    if os.path.exists(INDEX_DIR):
        vectorstore = FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(INDEX_DIR)

    # ---------------- BM25 ----------------
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 4

    return vectorstore, bm25

# ---------------------------------------------------------------------------
# Hybrid Retrieval Logic
# ---------------------------------------------------------------------------

def hybrid_retrieve(query, vectorstore, bm25):

    # Semantic search
    semantic_docs = vectorstore.similarity_search(query, k=4)

    # Keyword search (FIXED)
    keyword_docs = bm25.invoke(query)

    # Combine
    all_docs = semantic_docs + keyword_docs

    # Remove duplicates
    unique_docs = []
    seen = set()

    for doc in all_docs:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    return unique_docs[:6]

# ---------------------------------------------------------------------------
# LLM Setup
# ---------------------------------------------------------------------------

def get_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=API_KEY,
        temperature=0.3,
    )

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def get_prompt():

    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(HUMAN_PROMPT),
        ]
    )

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "bm25" not in st.session_state:
    st.session_state.bm25 = None

# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------

if not st.session_state.vectorstore:

    with st.spinner("Loading knowledge base..."):
        vs, bm25 = build_retrievers()

        st.session_state.vectorstore = vs
        st.session_state.bm25 = bm25

llm = get_llm()
prompt = get_prompt()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🎓 SSVPS Hybrid RAG Assistant")

# ---------------------------------------------------------------------------
# Chat Function
# ---------------------------------------------------------------------------

def handle_query(query):

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        docs = hybrid_retrieve(
            query,
            st.session_state.vectorstore,
            st.session_state.bm25,
        )

        context = "\n\n".join([d.page_content for d in docs])

        chain = prompt | llm

        response = chain.invoke(
            {"context": context, "question": query}
        )

        answer = response.content

        st.markdown(answer)

        with st.expander("Sources"):
            for i, d in enumerate(docs, 1):
                st.markdown(f"**Source {i}**")
                st.write(d.page_content[:300] + "...")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

# ---------------------------------------------------------------------------
# Chat History
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

if user_input := st.chat_input("Ask something about college"):

    handle_query(user_input)
    st.rerun()
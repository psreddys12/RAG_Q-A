import streamlit as st
import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
TARGET_URL = "https://generativeaimasters.in/"
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX = st.secrets["PINECONE_INDEX"]

st.set_page_config(
    page_title="Testing AI",
    layout="wide"
)

# ------------------------------------------------------------------
# INIT SERVICES
# ------------------------------------------------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY
)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)

# Pinecone v3+ connection (correct usage)
pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")

# Connect to existing Pinecone index only (do not create)

pinecone_index = pc.Index(PINECONE_INDEX)

vectorstore = PineconeVectorStore(
    index=pinecone_index,
    embedding=embeddings,
    text_key="text"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def clean_text(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return "\n".join(lines)

def extract_links(html: str):
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "javascript:", "#")):
            continue

        full = urljoin(TARGET_URL, href)
        parsed = urlparse(full)
        if "generativeaimasters.in" in parsed.netloc:
            links.add(full.rstrip("/"))

    return links

def crawl_website():
    visited = set()
    to_visit = [TARGET_URL.rstrip("/")]
    documents = []

    while to_visit and len(visited) < 200:
        url = to_visit.pop(0)
        if url in visited:
            continue

        visited.add(url)

        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                continue

            soup = BeautifulSoup(r.text, "html.parser")
            text = clean_text(soup.get_text())

            if len(text) > 200:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": url}
                    )
                )

            new_links = extract_links(r.text)
            for l in new_links:
                if l not in visited:
                    to_visit.append(l)

        except Exception:
            continue

    return documents

# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")


    if st.button("Index Website (One Time)"):
        with st.spinner("Crawling and indexing website..."):
            try:
                docs = crawl_website()
                st.write(f"Crawled {len(docs)} documents.")

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=300
                )

                chunks = splitter.split_documents(docs)
                st.write(f"Split into {len(chunks)} chunks.")

                PineconeVectorStore.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    index_name=PINECONE_INDEX
                )

                st.success(f"Indexed {len(chunks)} chunks into Pinecone")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ------------------------------------------------------------------
# UI HEADER
# ------------------------------------------------------------------
st.markdown("""
<h1 style="text-align:center;">Testing AI</h1>
<p style="text-align:center;color:gray;">
Ask anything about generativemasters
</p>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# CHAT DISPLAY
# ------------------------------------------------------------------
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ------------------------------------------------------------------
# RAG PROMPT
# ------------------------------------------------------------------
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert AI assistant for Generative AI Masters.\n\n"
     "Use the provided context to answer factual and informational questions accurately.\n"
     "If the user’s message is conversational (such as greetings, thanks, or general chat),\n"
     "respond politely and professionally without relying on the context.\n\n"
     "If the requested factual information is not available in the provided context,\n"
     "clearly and politely state that you do not currently have that information,\n"
     "and avoid making assumptions or providing unverified details.\n\n{context}"
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

def get_context(x):
    docs = retriever.invoke(x["input"])
    return format_docs(docs)


def get_chat_history(x):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    return st.session_state.chat_history

rag_chain = (
    {
        "context": RunnableLambda(get_context),
        "chat_history": get_chat_history,
        "input": lambda x: x["input"],
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)

# ------------------------------------------------------------------
# INPUT
# ------------------------------------------------------------------
prompt = st.chat_input("Ask a question about generativemasters...")

if prompt:
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": prompt})
            st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))
    st.rerun()

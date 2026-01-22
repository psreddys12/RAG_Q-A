import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import requests
import xml.etree.ElementTree as ET
import os
import shutil

# ---------------- CONFIG ----------------
TARGET_URL = "https://resolvetech.com"
SITEMAP_URL = "https://resolvetech.com/sitemap.xml"
VECTORSTORE_PATH = "data/faiss_index"
API_KEY = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="Resolve Tech AI", layout="wide")
st.title("Resolve Tech AI – Website Q&A Bot")

# ---------------- HELPERS ----------------
def clean_text(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def load_sitemap_urls():
    try:
        r = requests.get(SITEMAP_URL, timeout=10)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.text)
        ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        return [loc.text for loc in root.findall(".//ns:loc", ns)]
    except Exception:
        return []

def playwright_fetch(urls):
    docs = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        for url in urls:
            try:
                page.goto(url, timeout=20000)
                html = page.content()
                soup = BeautifulSoup(html, "html.parser")
                text = clean_text(soup.get_text())
                if len(text) > 50:
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": url}
                    ))
            except Exception:
                continue
        browser.close()
    return docs

def save_vectorstore(vs):
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vs.save_local(VECTORSTORE_PATH)

def load_vectorstore():
    if not os.path.exists(VECTORSTORE_PATH):
        return None
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=API_KEY
    )
    vs = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vs.as_retriever(search_kwargs={"k": 5})

# ---------------- SESSION ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = load_vectorstore()
if "index" not in st.session_state:
    st.session_state.index = False

# ---------------- SIDEBAR ----------------
with st.sidebar:
    if st.button("🔄 Index Website"):
        st.session_state.index = True
        st.session_state.retriever = None
        if os.path.exists(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH)

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []

# ---------------- INDEXING ----------------
if st.session_state.index:
    st.session_state.index = False

    with st.spinner("🔍 Loading sitemap…"):
        urls = load_sitemap_urls()
        st.info(f"Sitemap URLs found: {len(urls)}")

    if not urls:
        st.error("❌ Sitemap not available. Cannot index site.")
        st.stop()

    with st.spinner("🌐 Rendering pages with Playwright…"):
        docs = playwright_fetch(urls)
        st.info(f"Documents extracted: {len(docs)}")

    if not docs:
        st.error("❌ No readable text found even via browser rendering.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        st.error("❌ No chunks created.")
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=API_KEY
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    save_vectorstore(vectorstore)

    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    st.success("✅ Website indexed successfully")

# ---------------- CHAT ----------------
if st.session_state.retriever:
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    prompt = st.chat_input("Ask about Resolve Tech…")

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=API_KEY
        )

        def get_context(q):
            docs = st.session_state.retriever.invoke(q)
            return "\n\n".join(d.page_content for d in docs)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Answer ONLY using the provided context. "
             "If the answer is not present, say you do not know.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        chain = (
            {
                "context": RunnableLambda(lambda x: get_context(x["input"])),
                "chat_history": lambda x: st.session_state.chat_history,
                "input": RunnableLambda(lambda x: x["input"]),
            }
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        with st.chat_message("assistant"):
            answer = chain.invoke({"input": prompt})
            st.markdown(answer)

        st.session_state.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=answer)
        ])
else:
    st.info("👈 Click **Index Website** to begin.")

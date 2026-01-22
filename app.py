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
import requests
import os
import shutil
from urllib.parse import urljoin, urlparse

# ---------------- CONFIG ----------------
TARGET_URL = "https://resolvetech.com"
VECTORSTORE_PATH = "data/faiss_index"
API_KEY = st.secrets["GOOGLE_API_KEY"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

# ---------------- UI ----------------
st.set_page_config(page_title="Resolve Tech AI", layout="wide")
st.title("Resolve Tech AI – Website Q&A Bot")

# ---------------- HELPERS ----------------
def clean_text(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def extract_links(html: str, base_url: str) -> set:
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if href.startswith(("#", "mailto:", "javascript:")):
            continue
        full_url = urljoin(base_url, href).rstrip("/")
        parsed = urlparse(full_url)
        if parsed.netloc.endswith("resolvetech.com"):
            links.add(full_url)

    return links

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

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = load_vectorstore()

if "do_index" not in st.session_state:
    st.session_state.do_index = False

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Controls")

    if st.button("🔄 Index Website"):
        st.session_state.do_index = True
        st.session_state.retriever = None
        if os.path.exists(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH)

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []

# ---------------- INDEXING PIPELINE ----------------
if st.session_state.do_index:
    st.session_state.do_index = False

    # -------- STEP 1: CRAWL --------
    with st.spinner("🔍 Crawling website…"):
        visited = set()
        queue = [TARGET_URL]
        discovered = []
        max_pages = 200

        while queue and len(visited) < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue

            try:
                r = requests.get(
                    url,
                    headers=HEADERS,
                    timeout=15,
                    allow_redirects=True
                )

                if r.status_code != 200:
                    continue

                if "text/html" not in r.headers.get("Content-Type", ""):
                    continue

                visited.add(url)
                discovered.append(url)

                links = extract_links(r.text, url)
                for link in links:
                    if link not in visited and link not in queue:
                        queue.append(link)

            except Exception:
                continue

        st.success(f"✅ Discovered {len(discovered)} pages")

    # -------- STEP 2: LOAD CONTENT --------
    with st.spinner("📄 Loading page content…"):
        docs = []

        for url in discovered:
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                if r.status_code != 200:
                    continue

                soup = BeautifulSoup(r.text, "html.parser")
                text = clean_text(soup.get_text())

                # VERY IMPORTANT: low threshold
                if len(text) > 30:
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": url}
                        )
                    )

            except Exception:
                continue

        st.info(f"Documents loaded: {len(docs)}")

        if not docs:
            st.error("❌ No readable text found on the website.")
            st.stop()

    # -------- STEP 3: CHUNK --------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )

    chunks = splitter.split_documents(docs)
    st.info(f"Chunks created: {len(chunks)}")

    if not chunks:
        st.error("❌ No text chunks created. Indexing aborted.")
        st.stop()

    # -------- STEP 4: EMBEDDINGS + FAISS --------
    with st.spinner("🧠 Creating embeddings and FAISS index…"):
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=API_KEY
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)
        save_vectorstore(vectorstore)

        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        st.success("✅ Website indexed successfully")

# ---------------- CHAT INTERFACE ----------------
if st.session_state.retriever:
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    prompt = st.chat_input("Ask a question about Resolve Tech…")

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=API_KEY
        )

        def get_context(question: str) -> str:
            docs = st.session_state.retriever.invoke(question)
            return "\n\n".join(d.page_content for d in docs)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert on Resolve Tech. "
             "Answer ONLY using the provided context. "
             "If the answer is not in the context, say you do not know.\n\n{context}"),
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
    st.info("👈 Click **Index Website** to start crawling and indexing.")

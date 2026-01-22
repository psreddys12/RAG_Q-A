import streamlit as st
import requests
import glob
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PyPDF2 import PdfReader

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

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
TARGET_URL = "https://generativeaimasters.in/"
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX = st.secrets["PINECONE_INDEX"]

st.set_page_config(page_title="Generative AI Masters Chat", layout="wide")

# --------------------------------------------------
# SERVICES
# --------------------------------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(p.extract_text() or "" for p in reader.pages)

def load_pdfs():
    docs = []
    for pdf in glob.glob("data/pdfs/*.pdf"):
        text = extract_text_from_pdf(pdf)
        if len(text) > 200:
            docs.append(Document(page_content=text, metadata={"source": pdf}))
    return docs

def clean_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

def extract_links(html: str):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        full = urljoin(TARGET_URL, a["href"])
        if TARGET_URL in full:
            links.add(full.rstrip("/"))
    return links

def crawl_website():
    visited = set()
    queue = [TARGET_URL.rstrip("/")]
    docs = []

    while queue and len(visited) < 200:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            text = clean_text(soup.get_text())

            if len(text) > 200:
                docs.append(Document(page_content=text, metadata={"source": url}))

            queue.extend(extract_links(r.text) - visited)

        except Exception:
            continue

    return docs

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("Indexing")

    if st.button("Index PDFs"):
        docs = load_pdfs()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        ).split_documents(docs)

        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=PINECONE_INDEX
        )
        st.success("PDFs indexed")

    if st.button("Index Website"):
        docs = crawl_website()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        ).split_documents(docs)

        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=PINECONE_INDEX
        )
        st.success("Website indexed")

# --------------------------------------------------
# RAG
# --------------------------------------------------
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Use the context to answer accurately.\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

def get_context(x):
    docs = retriever.invoke(x["input"])
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {
        "context": RunnableLambda(get_context),
        "chat_history": lambda _: st.session_state.chat_history,
        "input": lambda x: x["input"],
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

# --------------------------------------------------
# CHAT UI
# --------------------------------------------------
st.title("Generative AI Masters Chat")

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

query = st.chat_input("Ask a question...")

if query:
    st.session_state.chat_history.append(HumanMessage(content=query))
    answer = rag_chain.invoke({"input": query})
    st.session_state.chat_history.append(AIMessage(content=answer))
    st.rerun()

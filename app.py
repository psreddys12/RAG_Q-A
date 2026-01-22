import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from bs4 import BeautifulSoup
import os
import shutil
import requests
from urllib.parse import urljoin, urlparse
import time

# --- Configuration ---
TARGET_URL = "https://resolvetech.com/"
API_KEY = st.secrets["GOOGLE_API_KEY"]
VECTORSTORE_PATH = "data/faiss_index"

st.set_page_config(page_title="Resolve Tech AI", page_icon="💻", layout="wide")

# --- Logic: Deep Crawler ---
def get_all_website_pages(base_url):
    """Exhaustively finds all internal links on the website."""
    visited = set()
    queue = [base_url.rstrip('/')]
    domain = urlparse(base_url).netloc

    while queue:
        url = queue.pop(0)
        if url in visited:
            continue
        
        try:
            # Respectful delay and User-Agent
            res = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebkit/537.36'})
            if res.status_code == 200:
                visited.add(url)
                soup = BeautifulSoup(res.text, 'html.parser')
                
                for a in soup.find_all('a', href=True):
                    link = urljoin(url, a['href']).split('#')[0].rstrip('/')
                    if urlparse(link).netloc == domain and link not in visited and link not in queue:
                        # Filter out non-html files
                        if not any(link.endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.zip']):
                            queue.append(link)
        except Exception:
            continue
            
    return list(visited)

def process_and_index(urls):
    """Scrapes content, splits into chunks, and saves FAISS index."""
    all_docs = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, url in enumerate(urls):
        status_text.text(f"Scraping ({i+1}/{len(urls)}): {url}")
        try:
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            
            # Clean HTML: remove nav, footer, scripts
            for extra in soup(['nav', 'footer', 'script', 'style', 'header']):
                extra.decompose()
            
            text = soup.get_text(separator=' ')
            clean_content = ' '.join(text.split())
            
            if len(clean_content) > 100:
                all_docs.append(Document(page_content=clean_content, metadata={"source": url}))
        except:
            continue
        progress_bar.progress((i + 1) / len(urls))

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    # Embed & Save
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

# --- Initialize State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar UI ---
with st.sidebar:
    st.header("Resolve Tech Admin")
    
    index_exists = os.path.exists(VECTORSTORE_PATH)
    
    if not index_exists:
        st.warning("No index found. Please crawl the website.")
        if st.button("🔍 Index Entire Website Now", use_container_width=True):
            with st.spinner("Discovering all pages... (This may take a minute)"):
                urls = get_all_website_pages(TARGET_URL)
                st.info(f"Found {len(urls)} pages. Starting deep crawl...")
                process_and_index(urls)
                st.success("Indexing Complete!")
                st.rerun()
    else:
        st.success("Website Indexed & Ready")
        if st.button("♻️ Re-crawl Website"):
            shutil.rmtree(VECTORSTORE_PATH)
            st.rerun()

# --- Main App Logic ---
st.title("💻 Resolve Tech AI Support")

if not os.path.exists(VECTORSTORE_PATH):
    st.info("👈 Please click the button in the sidebar to index the website content before chatting.")
else:
    # Load Index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Chat Setup
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)

    # 1. Contextualize Question
    context_q_system = "Given chat history and latest user question, make it a standalone question."
    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", context_q_system),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)

    # 2. Answer Chain
    qa_system = "You are a Resolve Tech expert. Use the context to answer. If unsure, say 'I don't know'.\n\nContext: {context}"
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # UI Chat Display
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    if user_input := st.chat_input("How can I help you today?"):
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing website data..."):
                response = rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.chat_history.append(AIMessage(content=answer))

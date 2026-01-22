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

# --- Config ---
TARGET_URL = "https://resolvetech.com/"
API_KEY = st.secrets["GOOGLE_API_KEY"]
DB_PATH = "data/faiss_index"

st.set_page_config(page_title="Resolve Tech AI", layout="wide")

# --- Deep Crawler Logic ---
def crawl_entire_site(url):
    domain = urlparse(url).netloc
    visited = set()
    to_visit = [url.rstrip('/')]
    documents = []
    
    progress_text = st.empty()
    
    while to_visit:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue
        
        try:
            res = requests.get(current_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            if res.status_code == 200:
                visited.add(current_url)
                soup = BeautifulSoup(res.text, 'html.parser')
                
                # Extract Text
                for s in soup(['script', 'style', 'nav', 'footer', 'header']):
                    s.decompose()
                
                clean_text = " ".join(soup.get_text(separator=" ").split())
                if len(clean_text) > 200:
                    documents.append(Document(page_content=clean_text, metadata={"source": current_url}))
                
                # Find More Links
                for a in soup.find_all('a', href=True):
                    full_link = urljoin(current_url, a['href']).split('#')[0].rstrip('/')
                    if urlparse(full_link).netloc == domain and full_link not in visited:
                        to_visit.append(full_link)
                
                progress_text.text(f"Analyzed: {len(visited)} pages... Found: {current_url}")
        except:
            continue
    return documents

# --- Sidebar Management ---
with st.sidebar:
    st.title("Resolve Tech Admin")
    if os.path.exists(DB_PATH):
        st.success("✅ Website Index Loaded")
        if st.button("Delete & Re-index"):
            shutil.rmtree("data")
            st.rerun()
    else:
        if st.button("🚀 Start Full Website Indexing"):
            with st.spinner("Crawling all pages..."):
                docs = crawl_entire_site(TARGET_URL)
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                final_docs = splitter.split_documents(docs)
                
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
                vectorstore = FAISS.from_documents(final_docs, embeddings)
                vectorstore.save_local(DB_PATH)
                st.success(f"Done! Indexed {len(docs)} pages.")
                st.rerun()

# --- Chat Interface ---
st.title("💻 Resolve Tech AI")

if not os.path.exists(DB_PATH):
    st.info("Please run the Indexing from the sidebar to start.")
else:
    # Load Retriever
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
    vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Setup LLM & Chains
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)

    # Prompt for re-writing the question based on history
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the history and latest question, make it a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

    # Final Answer Prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert for Resolve Tech. Use context to answer. Context: {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # Chat Session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask me anything about Resolve Tech"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate Answer
        history = [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) for m in st.session_state.messages[:-1]]
        
        with st.chat_message("assistant"):
            response = rag_chain.invoke({"input": prompt, "chat_history": history})
            st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

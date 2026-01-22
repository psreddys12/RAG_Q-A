import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from bs4 import BeautifulSoup

# --- Configuration & UI ---
TARGET_URL = "https://resolvetech.com/"
API_KEY = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="Resolve Tech AI", page_icon="💻", layout="wide")

# --- Custom CSS for ChatGPT-like interface ---
st.markdown("""
<style>
    /* Main container */
    .main {
        max-width: 1000px;
        margin: 0 auto;
    }
    
    /* Hide default header */
    [data-testid="stAppHeader"] {
        display: none;
    }
    
    /* Chat container styling */
    .chat-container {
        display: flex;
        flex-direction: column;
        height: 100vh;
        gap: 10px;
    }
    
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 20px;
    }
    
    .chat-input {
        padding: 20px;
        border-top: 1px solid #e5e5e5;
        background: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexing_done" not in st.session_state:
    st.session_state.indexing_done = False

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()
    
    process_btn = st.button("🔄 Index Website", use_container_width=True, key="index_btn")
    
    if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_btn"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    
    # Status indicator
    if "retriever" in st.session_state:
        st.success("✅ Website Indexed", icon="✅")
        st.caption("Ready to answer questions")
    else:
        st.warning("⚠️ Not Indexed", icon="⚠️")
        st.caption("Click 'Index Website' to start")

# --- RAG Indexing ---
if process_btn:
    with st.spinner("🔄 Processing entire website (this may take a minute)..."):
        try:
            loader = RecursiveUrlLoader(
                url=TARGET_URL,
                max_depth=5,
                extractor=lambda x: BeautifulSoup(x, "html.parser").get_text()
            )
            docs = loader.load()
            st.info(f"📄 Loaded {len(docs)} pages from the website")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(docs)
            st.info(f"✂️ Created {len(chunks)} text chunks")
            
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            st.session_state.indexing_done = True
            st.success("✅ Website indexed successfully!")
        except Exception as e:
            st.error(f"Error indexing website: {e}")

# --- Main Chat Interface ---
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="margin: 0; font-size: 2.5em;">💻 Resolve Tech AI</h1>
    <p style="margin: 10px 0; color: #666; font-size: 1.1em;">Ask anything about Resolve Tech Solutions</p>
</div>
""", unsafe_allow_html=True)

if "retriever" in st.session_state:
    # --- Chat Messages Display ---
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role, avatar="👤" if role == "user" else "🤖"):
                st.markdown(message.content)
    else:
        # Empty state
        st.markdown("""
        <div style="text-align: center; color: #999; padding: 60px 20px;">
            <p style="font-size: 1.2em; margin-bottom: 20px;">👋 Start a conversation</p>
            <p style="font-size: 0.9em;">Ask about cloud services, SAP solutions, or anything else about Resolve Tech</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Setup RAG Chain ---
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    qa_system_prompt = (
        "You are an expert for Resolve Tech Solutions. Use the following context to answer the question. "
        "If you don't know, say you don't know. Keep it professional.\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    retriever = st.session_state.retriever
    chat_history = st.session_state.chat_history
    
    rag_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["input"])),
            "chat_history": lambda x: chat_history,
            "input": lambda x: x["input"],
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # --- Chat Input (Bottom) ---
    st.markdown('<div class="chat-input">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([5, 1], gap="small")
    with col1:
        prompt = st.text_input(
            "Message...",
            placeholder="Ask about Cloud or SAP services...",
            key="chat_input",
            label_visibility="collapsed"
        )
    with col2:
        send_btn = st.button("Send", use_container_width=True, type="primary", key="send_btn")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Process Message ---
    if send_btn and prompt:
        # Add user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("⏳ Thinking..."):
                response = rag_chain.invoke({"input": prompt, "chat_history": st.session_state.chat_history})
            st.markdown(response)
        
        # Update history
        st.session_state.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=response),
        ])
        st.rerun()

else:
    # --- Initial State ---
    st.markdown("""
    <div style="text-align: center; padding: 100px 20px;">
        <p style="font-size: 1.1em; color: #666; margin-bottom: 30px;">
            👈 Click <strong>'Index Website'</strong> in the sidebar to get started
        </p>
        <div style="background: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <p style="margin: 10px 0;"><strong>How it works:</strong></p>
            <ul style="text-align: left; display: inline-block;">
                <li>📄 Crawls the entire Resolve Tech website</li>
                <li>🧠 Indexes all content for instant retrieval</li>
                <li>💬 Answers your questions based on website data</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


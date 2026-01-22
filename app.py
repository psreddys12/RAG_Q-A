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
import os

# --- Configuration & UI ---
TARGET_URL = "https://resolvetech.com/"
API_KEY = st.secrets["GOOGLE_API_KEY"]
VECTORSTORE_PATH = "data/faiss_index"  # Path to save vectorstore

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

# --- Helper Functions for Vectorstore Persistence ---
def save_vectorstore(vectorstore, path=VECTORSTORE_PATH):
    """Save FAISS vectorstore to disk"""
    try:
        os.makedirs(path, exist_ok=True)
        vectorstore.save_local(path)
        return True
    except Exception as e:
        st.error(f"Error saving vectorstore: {e}")
        return False

def load_vectorstore(path=VECTORSTORE_PATH):
    """Load FAISS vectorstore from disk"""
    try:
        if os.path.exists(path):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
            vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
            return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.warning(f"Could not load saved vectorstore: {e}")
    return None

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "current_team" not in st.session_state:
    st.session_state.current_team = "Resolve Tech AI"
if "retriever" not in st.session_state:
    # Try to load saved vectorstore first
    st.session_state.retriever = load_vectorstore()

# Define available teams
TEAMS = {
    "Resolve Tech AI": {"icon": "💻", "color": "#0066cc"},
    "Network Team": {"icon": "🌐", "color": "#FF6B6B"},
    "HR Team": {"icon": "👥", "color": "#4ECDC4"},
    "Servicenow Team": {"icon": "🔧", "color": "#FFE66D"},
    "Finance Team": {"icon": "💰", "color": "#95E1D3"},
}

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("""
    <style>
        .team-button {
            padding: 12px;
            margin: 8px 0;
            border-radius: 8px;
            border: 2px solid transparent;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s;
        }
        .team-button-active {
            border: 2px solid #0066cc;
            background-color: #f0f7ff;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("## 💬 Teams")
    st.divider()
    
    # New Chat button
    if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
        st.session_state.current_team = "Resolve Tech AI"
        st.session_state.chat_history[st.session_state.current_team] = []
        st.rerun()
    
    st.divider()
    
    # Team selection buttons
    for team_name, team_info in TEAMS.items():
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.markdown(f"<span style='font-size: 1.5em;'>{team_info['icon']}</span>", unsafe_allow_html=True)
        
        with col2:
            if st.button(team_name, use_container_width=True, key=f"team_{team_name}"):
                st.session_state.current_team = team_name
                # Initialize team chat history if not exists
                if team_name not in st.session_state.chat_history:
                    st.session_state.chat_history[team_name] = []
                st.rerun()
    
    st.divider()
    st.markdown("## ⚙️ Settings")
    st.divider()
    
    # Only show Index button if not yet indexed
    if st.session_state.retriever is None:
        process_btn = st.button("🔄 Index Website", use_container_width=True, key="index_btn")
    else:
        process_btn = False
        st.button("✅ Website Indexed", use_container_width=True, disabled=True, key="indexed_btn")
        if st.button("🔄 Re-Index Website", use_container_width=True, key="reindex_btn"):
            st.session_state.retriever = None
            st.rerun()
    
    # Clear current team chat
    if st.button("🗑️ Clear Current Chat", use_container_width=True, key="clear_btn"):
        if st.session_state.current_team in st.session_state.chat_history:
            st.session_state.chat_history[st.session_state.current_team] = []
        st.rerun()
    
    st.divider()
    
    # Status indicator
    if st.session_state.retriever is not None:
        st.success("✅ Website Indexed", icon="✅")
        st.caption("Ready to answer questions")
    else:
        st.warning("⚠️ Not Indexed", icon="⚠️")
        st.caption("Click 'Index Website' to start")

# --- RAG Indexing ---
if process_btn:
    with st.spinner("🔄 Processing entire website (this may take a few minutes)..."):
        try:
            st.info("📥 Loading pages from website...")
            
            # Use RecursiveUrlLoader with better settings
            loader = RecursiveUrlLoader(
                url=TARGET_URL,
                max_depth=3,  # Reduced depth to be more efficient
                prevent_outside=True,  # Stay within domain
                extractor=lambda x: BeautifulSoup(x, "html.parser").get_text(),
                continue_on_failure=True  # Continue even if some pages fail
            )
            docs = loader.load()
            
            if not docs:
                st.error("❌ No pages loaded from website. Please check the URL and try again.")
            else:
                st.success(f"✅ Loaded {len(docs)} pages from the website")
                
                st.info("✂️ Splitting documents into chunks...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=300,
                    separators=["\n\n", "\n", " ", ""]
                )
                chunks = text_splitter.split_documents(docs)
                st.success(f"✅ Created {len(chunks)} text chunks")
                
                st.info("🧠 Creating embeddings and building index...")
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
                vectorstore = FAISS.from_documents(chunks, embeddings)
                st.success("✅ Index created successfully")
                
                # Save vectorstore to disk
                if save_vectorstore(vectorstore):
                    st.success("💾 Vectorstore saved to disk")
                
                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                st.success("✅ Website indexed and ready to use!")
        except Exception as e:
            st.error(f"Error indexing website: {e}")

# --- Main Chat Interface ---
current_team = st.session_state.current_team
team_icon = TEAMS[current_team]["icon"]

st.markdown(f"""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="margin: 0; font-size: 2.5em;">{team_icon} {current_team}</h1>
    <p style="margin: 10px 0; color: #666; font-size: 1.1em;">Ask anything about Resolve Tech Solutions</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.retriever is not None:
    # --- Chat Messages Display ---
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    
    # Get current team's chat history
    current_team = st.session_state.current_team
    if current_team not in st.session_state.chat_history:
        st.session_state.chat_history[current_team] = []
    
    team_chat_history = st.session_state.chat_history[current_team]
    
    if team_chat_history:
        for message in team_chat_history:
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
    current_team = st.session_state.current_team
    
    # Get current team's chat history at chain creation time
    if current_team not in st.session_state.chat_history:
        st.session_state.chat_history[current_team] = []
    team_chat_history = st.session_state.chat_history[current_team]
    
    # Create dynamic RAG chain
    def get_context(input_dict):
        return format_docs(retriever.invoke(input_dict["input"]))
    
    def get_input(input_dict):
        return input_dict["input"]
    
    rag_chain = (
        {
            "context": RunnableLambda(get_context),
            "chat_history": lambda x: team_chat_history,
            "input": RunnableLambda(get_input),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # --- Chat Input (Bottom) ---
    st.markdown('<div class="chat-input">', unsafe_allow_html=True)
    
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1], gap="small")
        with col1:
            prompt = st.text_input(
                "Message...",
                placeholder="Ask about Cloud or SAP services...",
                key="chat_input",
                label_visibility="collapsed"
            )
        with col2:
            send_btn = st.form_submit_button("Send", use_container_width=True, type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Process Message ---
    if send_btn and prompt:
        # Add user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("⏳ Thinking..."):
                response = rag_chain.invoke({"input": prompt})
            st.markdown(response)
        
        # Update current team's history
        current_team = st.session_state.current_team
        st.session_state.chat_history[current_team].extend([
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


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
API_KEY = st.secrets["GOOGLE_API_KEY"]  # Get API key from Streamlit secrets

st.set_page_config(page_title="Resolve Tech AI", page_icon="💻", layout="wide")
st.title("💻 Resolve Tech Smart Assistant")
st.markdown("Ask questions about Resolve Tech Solutions' services")

# --- Initialize Session State for Memory ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Settings")
    process_btn = st.button("🔄 Index Website", use_container_width=True)
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    # Status indicator
    if "retriever" in st.session_state:
        st.success("✅ Website Indexed")
    else:
        st.info("ℹ️ Click 'Index Website' to start")

# --- RAG Indexing ---
if process_btn:
    with st.spinner("🔄 Processing entire website (this may take a minute)..."):
        try:
            # Use RecursiveUrlLoader to crawl the entire website
            loader = RecursiveUrlLoader(
                url=TARGET_URL,
                max_depth=5,  # Follow links up to 5 levels deep
                extractor=lambda x: BeautifulSoup(x, "html.parser").get_text()
            )
            docs = loader.load()
            st.info(f"📄 Loaded {len(docs)} pages from the website")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,  # Increased chunk size for better context
                chunk_overlap=300,  # Increased overlap for continuity
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(docs)
            st.info(f"✂️ Created {len(chunks)} text chunks")
            
            # Create embeddings and vector store
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Return top 5 results
            st.success("✅ Website indexed successfully!")
        except Exception as e:
            st.error(f"Error indexing website: {e}")

# --- The Conversational RAG Chain ---
if "retriever" in st.session_state:
    # 1. Setup LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)

    # 2. Contextualize Question (The "Memory" part)
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
    
    # Create a chain that rephrases questions using chat history
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
    
    # Function to handle chat history
    def get_context_from_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # 3. Answer Question
    qa_system_prompt = (
        "You are an expert for Resolve Tech Solutions. Use the following context to answer the question. "
        "If you don't know, say you don't know. Keep it professional.\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Capture retriever and chat history at chain creation time
    retriever = st.session_state.retriever
    chat_history = st.session_state.chat_history
    
    # Build the RAG chain using pipe operator - properly route context from retriever
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

    # --- Chat History Display ---
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("💬 Conversation History")
        for message in st.session_state.chat_history:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)

    # --- Search Box on Main Page ---
    st.markdown("---")
    st.subheader("🔍 Ask a Question")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        prompt = st.text_input("Search...", placeholder="Ask about Cloud or SAP services...", key="search_input")
    with col2:
        search_btn = st.button("Search", use_container_width=True)
    
    if search_btn and prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
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
    st.info("👈 Click **Index Website** in the sidebar to get started!")


import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Setup Streamlit UI ---
st.set_page_config(page_title="Gemini Website RAG Agent", layout="wide")
st.title("🤖 Website AI Agent")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Enter Website URL", placeholder="https://example.com")
    api_key = st.text_input("Gemini API Key", type="password")
    process_btn = st.button("Process Website")

# --- 2. RAG Logic ---
if process_btn and website_url and api_key:
    try:
        # Load data from the web
        loader = WebBaseLoader(website_url)
        docs = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Create Vector Store (using FAISS)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        
        # Save to session state to avoid re-processing
        st.session_state.retriever = vectorstore.as_retriever()
        st.success("Website indexed successfully!")
    except Exception as e:
        st.error(f"Error: {e}")

# --- 3. Chat Interface ---
if "retriever" in st.session_state:
    user_input = st.chat_input("Ask something about the website...")
    
    if user_input:
        # Define the LLM (Gemini 2.5 Pro)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key)

        # Define the Prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say you don't know.\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Create Chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(st.session_state.retriever, question_answer_chain)

        # Get response
        response = rag_chain.invoke({"input": user_input})
        
        with st.chat_message("assistant"):
            st.write(response["answer"])

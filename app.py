import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration & UI (Same as before) ---
TARGET_URL = "https://resolvetech.com/"
st.set_page_config(page_title="Resolve Tech AI", page_icon="💻")
st.title("💻 Resolve Tech Smart Assistant")

with st.sidebar:
    api_key = st.text_input("AIzaSyBTqfh8bi0ctEEvtvWk0bdoTy0FVpDKkzI", type="password")
    process_btn = st.button("Index Website")
    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.rerun()

# --- Initialize Session State for Memory ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- RAG Indexing (Same Logic) ---
if process_btn and api_key:
    with st.spinner("Processing..."):
        loader = WebBaseLoader(TARGET_URL)
        loader.requests_kwargs = {'headers': {'User-Agent': 'Mozilla/5.0'}}
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.session_state.retriever = vectorstore.as_retriever()
        st.success("Website Indexed!")

# --- The Conversational RAG Chain ---
if "retriever" in st.session_state:
    # 1. Setup LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

    # 2. Contextualize Question (The "Memory" part)
    # This rephrases the user's question to be standalone
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
    
    # Build the RAG chain using pipe operator
    rag_chain = (
        {
            "context": st.session_state.retriever | RunnableLambda(get_context_from_docs),
            "chat_history": RunnableLambda(lambda x: st.session_state.chat_history),
            "input": RunnablePassthrough(),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # --- Chat Display ---
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    if prompt := st.chat_input("Ask about their Cloud or SAP services..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = rag_chain.invoke({"input": prompt, "chat_history": st.session_state.chat_history})
            st.markdown(response)
            
        # Update history
        st.session_state.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=response),
        ])

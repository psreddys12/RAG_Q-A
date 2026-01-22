import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from pypdf import PdfReader
import os
import shutil

# ---------------- CONFIG ----------------
VECTORSTORE_PATH = "data/faiss_index"
UPLOAD_DIR = "data/uploads"
API_KEY = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="Hybrid RAG – PDF Chatbot", layout="wide")
st.title("📄 Hybrid RAG Chatbot (PDF Based)")

# ---------------- HELPERS ----------------
def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

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

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📂 Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🔄 Index PDFs"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
            st.stop()

        # Reset index
        st.session_state.retriever = None
        if os.path.exists(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH)

        os.makedirs(UPLOAD_DIR, exist_ok=True)

        docs = []

        with st.spinner("📄 Reading PDFs…"):
            for file in uploaded_files:
                file_path = os.path.join(UPLOAD_DIR, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.read())

                text = extract_text_from_pdf(file_path)
                if len(text) > 50:
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": file.name}
                    ))

        st.info(f"Documents loaded: {len(docs)}")

        if not docs:
            st.error("No readable text found in PDFs.")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)
        st.info(f"Chunks created: {len(chunks)}")

        if not chunks:
            st.error("No chunks created.")
            st.stop()

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=API_KEY
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)
        save_vectorstore(vectorstore)

        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        st.success("✅ PDFs indexed successfully")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []

# ---------------- CHAT ----------------
if st.session_state.retriever:
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    prompt = st.chat_input("Ask a question from your PDFs…")

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
             "Answer ONLY using the provided context from documents. "
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
    st.info("👈 Upload PDFs and click **Index PDFs** to begin.")

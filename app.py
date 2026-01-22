import glob
from PyPDF2 import PdfReader
# ------------------------------------------------------------------
# PDF HELPERS
# ------------------------------------------------------------------
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    with st.sidebar:
        st.header("Controls")

        if st.button("Index PDFs (One Time)"):
            with st.spinner("Extracting and indexing PDFs..."):
                try:
                    docs = load_pdfs_from_folder()
                    st.write(f"Loaded {len(docs)} PDF documents.")

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=300
                    )

                    chunks = splitter.split_documents(docs)
                    st.write(f"Split into {len(chunks)} chunks.")

                    PineconeVectorStore.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        index_name=PINECONE_INDEX
                    )

                    st.success(f"Indexed {len(chunks)} PDF chunks into Pinecone")
                except Exception as e:
                    st.error(f"PDF Indexing failed: {e}")

        if st.button("Index Website (One Time)"):
            with st.spinner("Crawling and indexing website..."):
                try:
                    docs = crawl_website()
                    st.write(f"Crawled {len(docs)} documents.")

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=300
                    )

                    chunks = splitter.split_documents(docs)
                    st.write(f"Split into {len(chunks)} chunks.")

                    PineconeVectorStore.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        index_name=PINECONE_INDEX
                    )

                    st.success(f"Indexed {len(chunks)} chunks into Pinecone")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY
)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)

# Pinecone v3+ connection (correct usage)
pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")

# Connect to existing Pinecone index only (do not create)

pinecone_index = pc.Index(PINECONE_INDEX)

vectorstore = PineconeVectorStore(
    index=pinecone_index,
    embedding=embeddings,
    text_key="text"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def clean_text(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return "\n".join(lines)

def extract_links(html: str):
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "javascript:", "#")):
            continue

        full = urljoin(TARGET_URL, href)
        parsed = urlparse(full)
        if "generativeaimasters.in" in parsed.netloc:
            links.add(full.rstrip("/"))

    return links

def crawl_website():
    visited = set()
    to_visit = [TARGET_URL.rstrip("/")]
    documents = []

    while to_visit and len(visited) < 200:
        url = to_visit.pop(0)
        if url in visited:
            continue

        visited.add(url)

        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                continue

            soup = BeautifulSoup(r.text, "html.parser")
            text = clean_text(soup.get_text())

            if len(text) > 200:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": url}
                    )
                )

            new_links = extract_links(r.text)
            for l in new_links:
                if l not in visited:
                    to_visit.append(l)

        except Exception:
            continue

    return documents

# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:

        if st.button("Index PDFs (One Time)"):
            with st.spinner("Extracting and indexing PDFs..."):
                try:
                    docs = load_pdfs_from_folder()
                    st.write(f"Loaded {len(docs)} PDF documents.")

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=300
                    )

                    chunks = splitter.split_documents(docs)
                    st.write(f"Split into {len(chunks)} chunks.")

                    PineconeVectorStore.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        index_name=PINECONE_INDEX
                    )

                    st.success(f"Indexed {len(chunks)} PDF chunks into Pinecone")
                except Exception as e:
                    st.error(f"PDF Indexing failed: {e}")
    st.header("Controls")


    if st.button("Index Website (One Time)"):
        with st.spinner("Crawling and indexing website..."):
            try:
                docs = crawl_website()
                st.write(f"Crawled {len(docs)} documents.")

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=300
                )

                chunks = splitter.split_documents(docs)
                st.write(f"Split into {len(chunks)} chunks.")

                PineconeVectorStore.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    index_name=PINECONE_INDEX
                )

                st.success(f"Indexed {len(chunks)} chunks into Pinecone")
            except Exception as e:
                st.error(f"Indexing failed: {e}")


    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ------------------------------------------------------------------
# UI HEADER & STYLES (Top-level, not indented)
# ------------------------------------------------------------------
st.markdown("""
<style>
.main-header {
    text-align: center;

        st.header("Controls")

        if st.button("Index PDFs (One Time)"):
            with st.spinner("Extracting and indexing PDFs..."):
                try:
                    docs = load_pdfs_from_folder()
                    st.write(f"Loaded {len(docs)} PDF documents.")

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=300
                    )

                    chunks = splitter.split_documents(docs)
                    st.write(f"Split into {len(chunks)} chunks.")

                    PineconeVectorStore.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        index_name=PINECONE_INDEX
                    )

                    st.success(f"Indexed {len(chunks)} PDF chunks into Pinecone")
                except Exception as e:
                    st.error(f"PDF Indexing failed: {e}")

        if st.button("Index Website (One Time)"):
            with st.spinner("Crawling and indexing website..."):
                try:
                    docs = crawl_website()
                    st.write(f"Crawled {len(docs)} documents.")

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=300
                    )

                    chunks = splitter.split_documents(docs)
                    st.write(f"Split into {len(chunks)} chunks.")

                    PineconeVectorStore.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        index_name=PINECONE_INDEX
                    )

                    st.success(f"Indexed {len(chunks)} chunks into Pinecone")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
# CHAT DISPLAY (Stylish with Avatars, only once)
# ------------------------------------------------------------------
user_logo = "<span style='font-size:2rem;'>🧑‍💻</span>"
bot_logo = "<span style='font-size:2rem;'>🤖</span>"

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    logo = user_logo if role == "user" else bot_logo
    bubble_class = f"chat-bubble {role}"
    st.markdown(f"""
    <div class="{bubble_class}">
        <div class="chat-avatar">{logo}</div>
        <div class="chat-content">{msg.content}</div>
    </div>
    """, unsafe_allow_html=True)
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

qa_prompt = ChatPromptTemplate.from_messages([
     ("system",
      "You are an expert AI assistant for Generative AI Masters.\n\n"
      "Use the provided context to answer factual and informational questions accurately.\n"
      "If the user’s message is conversational (such as greetings, thanks, or general chat),\n"
      "respond politely and professionally without relying on the context.\n\n"
      "If the requested factual information is not available in the provided context,\n"
      "clearly and politely state that you do not currently have that information,\n"
      "and avoid making assumptions or providing unverified details.\n\n{context}"
     ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

def get_context(x):
    docs = retriever.invoke(x["input"])
    return format_docs(docs)


def get_chat_history(x):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    return st.session_state.chat_history

rag_chain = (
    {
        "context": RunnableLambda(get_context),
        "chat_history": get_chat_history,
        "input": lambda x: x["input"],
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)


# ------------------------------------------------------------------
# INPUT
# ------------------------------------------------------------------
prompt = st.chat_input("Type your message and press Enter...")

if prompt:
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": prompt})
            st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))
    st.rerun()

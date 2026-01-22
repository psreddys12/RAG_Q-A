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
from langchain_core.documents import Document
from bs4 import BeautifulSoup
import os
import shutil
import requests
from urllib.parse import urljoin, urlparse

# --- Configuration & UI ---
TARGET_URL = "https://rpachallenge.com/"
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

def clean_text(text):
    """Clean and normalize text"""
    # Remove extra whitespace
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    return '\n'.join(lines)

def extract_links_from_html(html_content):
    """Extract all links from HTML"""
    links = set()
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for link in soup.find_all('a', href=True):
        href = link.get('href', '').strip()
        if href:
            # Handle relative URLs
            if href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                continue
            full_url = urljoin(TARGET_URL, href)
            # Only keep links from same domain
            parsed = urlparse(full_url)
            if 'rpachallenge.com' in parsed.netloc:
                links.add(full_url.rstrip('/'))
    
    return links

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    # Try to load saved vectorstore first
    st.session_state.retriever = load_vectorstore()
if "test_crawler" not in st.session_state:
    st.session_state.test_crawler = False

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("## 💬 Chat")
    st.divider()
    
    # New Chat button
    if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
        st.session_state.chat_history = []
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
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_btn"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Debug button to test crawling
    if st.button("🔍 Test Crawler", use_container_width=True, key="test_crawler_btn"):
        st.session_state.test_crawler = True
        st.rerun()
    
    # Force clear button
    if st.button("🔥 Force Clear All", use_container_width=True, key="force_clear_btn"):
        st.session_state.retriever = None
        st.session_state.chat_history = []
        if os.path.exists(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH)
        st.success("✅ All data cleared! Please refresh the page.")
        st.rerun()
    
    st.divider()
    
    # Status indicator
    if st.session_state.retriever is not None:
        st.success("✅ Website Indexed", icon="✅")
        st.caption("Ready to answer questions")
        
        # Test retrieval
        if st.button("🧪 Test Search", use_container_width=True, key="test_search_btn"):
            test_query = "leadership team company"
            try:
                test_results = st.session_state.retriever.invoke(test_query)
                st.info(f"Test search for '{test_query}' found {len(test_results)} results")
                for i, doc in enumerate(test_results, 1):
                    st.caption(f"{i}. {doc.metadata.get('source', 'Unknown')}")
                    st.write(doc.page_content[:200] + "...")
            except Exception as e:
                st.error(f"Search test failed: {e}")
    else:
        st.warning("⚠️ Not Indexed", icon="⚠️")
        st.caption("Click 'Index Website' to start")

# --- Test Crawler (Debug) ---
if st.session_state.test_crawler:
    st.session_state.test_crawler = False
    st.warning("Testing web crawler - scanning for all links...")
    try:
        visited = set()
        to_visit = [TARGET_URL]
        all_links = set()
        
        st.info(f"Discovering links from {TARGET_URL}...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Crawl and discover links
        while to_visit and len(visited) < 100:  # Limit to 100 pages for testing
            url = to_visit.pop(0)
            if url in visited:
                continue
            
            visited.add(url)
            status_text.text(f"Discovered {len(all_links)} unique links | Visited {len(visited)} pages")
            progress_bar.progress(min(len(visited) / 50, 1.0))
            
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    # Extract links from this page
                    new_links = extract_links_from_html(response.text)
                    all_links.update(new_links)
                    
                    # Add new links to queue
                    for link in new_links:
                        if link not in visited and link not in to_visit:
                            to_visit.append(link)
            except Exception as e:
                st.caption(f"⚠️ Error fetching {url}: {str(e)[:50]}")
        
        st.success(f"✅ Found {len(all_links)} unique pages:")
        with st.expander("📋 All Pages Found", expanded=True):
            for i, link in enumerate(sorted(all_links), 1):
                st.write(f"{i}. {link}")
                
    except Exception as e:
        st.error(f"Crawler error: {e}")
        import traceback
        st.error(traceback.format_exc())

# --- RAG Indexing ---
if process_btn:
    with st.spinner("🔄 Processing entire website (this may take a few minutes)..."):
        try:
            st.info("🗑️ Clearing old index...")
            if os.path.exists(VECTORSTORE_PATH):
                shutil.rmtree(VECTORSTORE_PATH)
                st.success("Old index cleared")
            
            st.info("🔍 Discovering all pages on the website...")
            
            # Step 1: Discover all links using BFS
            visited = set()
            to_visit = [TARGET_URL.rstrip('/')]
            all_links = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            errors_container = st.container()
            
            page_count = 0
            max_pages = 200
            
            while to_visit and len(visited) < max_pages:
                url = to_visit.pop(0)
                
                # Skip if already visited
                if url in visited:
                    continue
                
                visited.add(url)
                all_links.append(url)
                page_count += 1
                
                status_text.text(f"Discovered {page_count} pages | Queue: {len(to_visit)} | Current: {url}")
                progress_bar.progress(min(page_count / 50, 1.0))
                
                try:
                    response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
                    if response.status_code == 200:
                        # Extract links from this page
                        new_links = extract_links_from_html(response.text)
                        
                        for link in new_links:
                            clean_link = link.rstrip('/')
                            if clean_link not in visited and clean_link not in to_visit:
                                to_visit.append(clean_link)
                    else:
                        st.warning(f"⚠️ Failed to fetch {url}: Status {response.status_code}")
                        
                except requests.exceptions.Timeout:
                    st.warning(f"⚠️ Timeout: {url}")
                except requests.exceptions.RequestException as e:
                    st.warning(f"⚠️ Error fetching {url}: {str(e)[:50]}")
                except Exception as e:
                    st.warning(f"⚠️ Unexpected error with {url}: {str(e)[:50]}")
            
            st.success(f"✅ Discovered {len(all_links)} unique pages in {page_count} attempts")
            
            # Show all discovered links
            with st.expander("🔗 All Discovered Pages", expanded=False):
                for idx, link in enumerate(all_links, 1):
                    st.caption(f"{idx}. {link}")
            
            # Step 2: Load content from all discovered pages
            st.info(f"📥 Loading content from {len(all_links)} pages...")
            docs = []
            failed_urls = []
            
            for idx, url in enumerate(all_links):
                progress_bar.progress(min((idx + 1) / len(all_links), 1.0))
                status_text.text(f"Loading {idx + 1}/{len(all_links)}: {url}")
                
                try:
                    response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
                    if response.status_code == 200:
                        # Extract text from HTML
                        text = BeautifulSoup(response.text, "html.parser").get_text()
                        text = clean_text(text)  # Clean the text
                        
                        if text.strip() and len(text.strip()) > 50:  # Only add if meaningful content
                            doc = Document(
                                page_content=text,
                                metadata={"source": url}
                            )
                            docs.append(doc)
                        else:
                            st.caption(f"⚠️ Skipped {url} (too short: {len(text)} chars)")
                    else:
                        failed_urls.append((url, response.status_code))
                        
                except Exception as e:
                    failed_urls.append((url, str(e)))
            
            if failed_urls:
                st.warning(f"⚠️ Failed to load {len(failed_urls)} pages")
            
            st.success(f"✅ Loaded {len(docs)} pages with content")
            
            # Show details of loaded pages
            with st.expander("📋 Pages Indexed Successfully", expanded=True):
                st.write(f"**Total Pages Indexed: {len(docs)}**")
                if len(docs) > 0:
                    for i, doc in enumerate(docs, 1):
                        url = doc.metadata.get('source', 'Unknown')
                        content_size = len(doc.page_content)
                        preview = doc.page_content[:80].replace('\n', ' ').strip()
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"{i}. {url}")
                            st.caption(f"Preview: {preview}...")
                        with col2:
                            st.metric("Size", f"{content_size:,}")
                else:
                    st.error("❌ No pages were loaded!")
            
            if not docs:
                st.error("❌ No content loaded. Please check the website.")
            else:
                st.info(f"📄 Total content size: {sum(len(d.page_content) for d in docs):,} characters")
                
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
                st.info("💾 Saving vectorstore to disk...")
                if save_vectorstore(vectorstore):
                    st.success("✅ Vectorstore saved to disk!")
                    # Verify files were created
                    if os.path.exists(VECTORSTORE_PATH):
                        files = os.listdir(VECTORSTORE_PATH)
                        st.success(f"✅ Vectorstore files created: {', '.join(files)}")
                else:
                    st.error("❌ Failed to save vectorstore")
                
                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                st.success("✅ Website indexed and ready to use!")
                
        except Exception as e:
            st.error(f"Error indexing website: {e}")
            import traceback
            st.error(traceback.format_exc())

# --- Main Chat Interface ---
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="margin: 0; font-size: 2.5em;">💻 Resolve Tech AI</h1>
    <p style="margin: 10px 0; color: #666; font-size: 1.1em;">Ask anything about Resolve Tech Solutions</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.retriever is not None:
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
    
    # Create dynamic RAG chain
    def get_context(input_dict):
        return format_docs(retriever.invoke(input_dict["input"]))
    
    def get_input(input_dict):
        return input_dict["input"]
    
    rag_chain = (
        {
            "context": RunnableLambda(get_context),
            "chat_history": lambda x: chat_history,
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


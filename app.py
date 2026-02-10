"""
app.py
======
Streamlit RAG application with intelligent FAISS caching.

Architecture:
-------------
- UI Layer (this file): Handles Streamlit interface
- Vector Store Layer (vector_store.py): Handles embeddings & caching
- LLM Layer: Gemini API for answer generation

Caching Strategy:
-----------------
1. @st.cache_resource: Cache embeddings model (loaded once)
2. Disk persistence: FAISS indexes cached in vector_store.py
3. Session state: Track current PDF to invalidate cache on change
"""

import streamlit as st
from pypdf import PdfReader
import pandas as pd
import base64
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI

# Import our vector store module
import vector_store


# ============================================================================
# STREAMLIT CACHING
# ============================================================================

@st.cache_resource
def load_embeddings_model():
    """
    Load embeddings model ONCE and cache it.
    
    Uses @st.cache_resource because:
    - HuggingFace model is a non-serializable resource
    - Should be shared across all reruns
    - Should persist in memory
    
    Returns:
        HuggingFaceEmbeddings: Cached embeddings model
    """
    return vector_store.get_embeddings_model()


# ============================================================================
# PDF PROCESSING
# ============================================================================

def extract_pdf_text(pdf_file) -> str:
    """
    Extract text from uploaded PDF.
    
    Args:
        pdf_file: Streamlit UploadedFile object
        
    Returns:
        str: Extracted text
    """
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
    
    return text


# ============================================================================
# LLM INTEGRATION
# ============================================================================

def get_answer_from_context(context: str, question: str, api_key: str) -> str:
    """
    Generate answer using Gemini API.
    
    Args:
        context: Relevant text chunks from PDF
        question: User's question
        api_key: Google API key
        
    Returns:
        str: Generated answer
    """
    prompt = f"""Answer the question based on the provided context. Be detailed and accurate.

Context:
{context}

Question:
{question}

Answer:"""
    
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )
    
    response = model.invoke(prompt)
    return response.content


# ============================================================================
# UI STYLING
# ============================================================================

def load_css():
    """Apply custom CSS styling."""
    st.markdown("""
    <style>
        /* Chat messages */
        .chat-message {
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            display: flex;
            animation: fadeIn 0.5s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .chat-message.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-left: 5px solid #764ba2;
        }
        
        .chat-message.bot {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border-left: 5px solid #f5576c;
        }
        
        .chat-message .avatar {
            width: 15%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .chat-message .avatar img {
            max-width: 60px;
            max-height: 60px;
            border-radius: 50%;
            border: 3px solid white;
        }
        
        .chat-message .message {
            width: 85%;
            padding: 0 1.5rem;
            color: #fff;
            line-height: 1.6;
        }
        
        .chat-message .info {
            font-size: 0.85rem;
            margin-top: 0.8rem;
            opacity: 0.9;
        }
        
        /* Stats card */
        .stats-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .stats-card h3 {
            margin: 0;
            font-size: 1.5rem;
        }
        
        .stats-card p {
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Page config
    st.set_page_config(
        page_title="Chat with PDF",
        page_icon="📄",
        layout="wide"
    )
    
    # Load CSS
    load_css()
    
    # Header
    st.title("Chat with PDF")
    
    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'current_pdf_name' not in st.session_state:
        st.session_state.current_pdf_name = None
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    # Load embeddings model (cached)
    embeddings_model = load_embeddings_model()
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "🔑 Google API Key:",
            type="password",
            help="Required for answer generation (Gemini)"
        )
        st.markdown("[Get API Key](https://ai.google.dev/)")
        
        st.markdown("---")
        st.title("📁 Upload PDF")
        
        # PDF Upload
        pdf_file = st.file_uploader(
            "Upload your PDF",
            type=['pdf'],
            help="Max 200MB, single PDF at a time"
        )
        
        # Process PDF button
        if pdf_file:
            st.success(f"✅ {pdf_file.name}")
            st.text(f"Size: {pdf_file.size / (1024*1024):.2f} MB")
            
            # Check if this is a NEW pdf (different from current)
            pdf_changed = st.session_state.current_pdf_name != pdf_file.name
            
            if st.button("🚀 Process PDF", use_container_width=True):
                with st.spinner("Processing PDF..."):
                    # Extract text
                    raw_text = extract_pdf_text(pdf_file)
                    
                    if not raw_text.strip():
                        st.error("❌ No text found in PDF")
                        return
                    
                    # Get or create vector store (with intelligent caching)
                    st.session_state.vector_store = vector_store.get_or_create_vector_store(
                        pdf_file,
                        raw_text,
                        embeddings_model
                    )
                    
                    st.session_state.current_pdf_name = pdf_file.name
                    
                    # Clear old conversations when new PDF uploaded
                    if pdf_changed:
                        st.session_state.conversation_history = []
                    
                    st.success("✅ PDF processed successfully!")
                    st.rerun()
        
        st.markdown("---")
        st.title("🔧 Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state.conversation_history = []
                st.success("✅ Chat cleared")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                vector_store.clear_cache()
                st.session_state.vector_store = None
                st.session_state.current_pdf_name = None
                st.success("✅ Cache cleared")
                st.rerun()
        
        st.markdown("---")
        
        # Stats
        if len(st.session_state.conversation_history) > 0:
            st.markdown(f"""
                <div class="stats-card">
                    <h3>{len(st.session_state.conversation_history)}</h3>
                    <p>Questions Asked</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Download conversation
            df = pd.DataFrame(
                st.session_state.conversation_history,
                columns=["Question", "Answer", "Timestamp", "PDF"]
            )
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(
                f'<a href="data:file/csv;base64,{b64}" download="conversation.csv" '
                f'style="text-decoration:none;"><button style="width:100%;padding:0.5rem;'
                f'background:#667eea;color:white;border:none;border-radius:8px;'
                f'cursor:pointer;">📥 Download Chat</button></a>',
                unsafe_allow_html=True
            )
        
        # Info
        with st.expander("ℹ️ How It Works"):
            st.markdown("""
            **Caching Strategy:**
            1. Upload PDF → Extract text
            2. Generate embeddings (ONCE)
            3. Save FAISS index to disk
            4. Future queries reuse cached embeddings
            
            **No Regeneration:**
            - Same PDF = Load from cache
            - New PDF = Generate & cache
            - No API rate limits on embeddings
            
            **Models:**
            - Embeddings: `all-MiniLM-L6-v2` (local)
            - Chat: `gemini-2.5-flash` (API)
            """)
    
    # Main area
    st.markdown("### 🔍 Ask Your Question")
    
    # Question input
    user_question = st.text_input(
        "Type your question:",
        placeholder="What would you like to know?",
        label_visibility="collapsed"
    )
    
    # Handle question
    if user_question and user_question.strip():
        # Validation
        if not api_key:
            st.warning("⚠️ Please enter your Google API key")
            return
        
        if st.session_state.vector_store is None:
            st.warning("⚠️ Please upload and process a PDF first")
            return
        
        with st.spinner("🤔 Thinking..."):
            try:
                # Search in vector store (NO regeneration, uses cached embeddings)
                relevant_docs = vector_store.similarity_search(
                    st.session_state.vector_store,
                    user_question,
                    k=4
                )
                
                # Combine context
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # Get answer from Gemini
                answer = get_answer_from_context(context, user_question, api_key)
                
                # Save to history
                st.session_state.conversation_history.append((
                    user_question,
                    answer,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    st.session_state.current_pdf_name
                ))
                
                # Display
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="avatar">
                        <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                    </div>
                    <div class="message">
                        <strong>You:</strong><br>{user_question}
                        <div class="info">📅 {datetime.now().strftime('%H:%M:%S')}</div>
                    </div>
                </div>
                <div class="chat-message bot">
                    <div class="avatar">
                        <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
                    </div>
                    <div class="message">
                        <strong>Assistant:</strong><br>{answer}
                        <div class="info">🤖 Gemini 2.5 Flash | 📄 {st.session_state.current_pdf_name}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.success("✅ Answer generated!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Display conversation history
    if len(st.session_state.conversation_history) > 0:
        st.markdown("---")
        st.subheader("💬 Conversation History")
        
        for i, (question, answer, timestamp, pdf_name) in enumerate(
            reversed(st.session_state.conversation_history)
        ):
            with st.expander(
                f"Q{len(st.session_state.conversation_history)-i}: {question[:60]}...",
                expanded=(i == 0)
            ):
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:** {answer}")
                st.caption(f"🕐 {timestamp} | 📄 {pdf_name}")


if __name__ == "__main__":
    main()
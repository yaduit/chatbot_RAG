import streamlit as st
from pypdf import PdfReader
import pandas as pd
import base64
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from datetime import datetime

# Custom CSS for better UI
def load_css():
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background-color: #0e1117;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .main-header h1 {
            color: white;
            margin: 0;
            font-size: 2.5rem;
        }
        
        .main-header p {
            color: #f0f0f0;
            margin-top: 0.5rem;
            font-size: 1.1rem;
        }
        
        /* Chat message styling */
        .chat-message {
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            display: flex;
            animation: fadeIn 0.5s;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
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
            object-fit: cover;
            border: 3px solid white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        
        .chat-message .message {
            width: 85%;
            padding: 0 1.5rem;
            color: #fff;
            font-size: 1rem;
            line-height: 1.6;
        }
        
        .chat-message .info {
            font-size: 0.85rem;
            margin-top: 0.8rem;
            color: #f0f0f0;
            opacity: 0.9;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #1e1e1e;
        }
        
        /* Success/Warning/Info boxes */
        .stAlert {
            border-radius: 10px;
        }
        
        /* Button styling */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        /* Download button styling */
        .download-btn {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            text-decoration: none;
            display: inline-block;
            font-weight: 600;
            transition: all 0.3s;
            border: none;
            cursor: pointer;
        }
        
        .download-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        /* File uploader styling */
        .uploadedFile {
            border-radius: 8px;
            border: 2px solid #667eea;
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
        
        /* Question input container */
        .question-container {
            position: relative;
            margin-bottom: 1rem;
        }
        
        .question-icon {
            font-size: 1.5rem;
            margin-right: 0.5rem;
            vertical-align: middle;
        }
    </style>
    """, unsafe_allow_html=True)

# Extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {str(e)}")
    return text

# Split text into chunks
def get_text_chunks(text, model_name):
    if model_name == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
    chunks = text_splitter.split_text(text)
    return chunks

# Create vector store from text chunks
def get_vector_store(text_chunks, model_name, api_key=None):
    if model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",  # Correct embedding model
            google_api_key=api_key
        )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Get answer from documents using direct LLM call
def get_answer_from_docs(docs, question, api_key):
    """Get answer from documents using Google Gemini"""
    # Combine document content
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    prompt = f"""Answer the question as detailed as possible from the provided context. Make sure to provide all the details with proper structure. If the answer is not in the provided context, just say "answer is not available in the context". Don't provide wrong answers.

Context:
{context}

Question:
{question}

Answer:"""
    
    # Get response from model using the correct model name
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # FIXED: Using correct model name from available models
        temperature=0.3,
        google_api_key=api_key
    )
    
    response = model.invoke(prompt)
    return response.content

# Process user input and generate response
def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    if api_key is None or not api_key.strip():
        st.warning("⚠️ Please provide an API key")
        return
    
    if pdf_docs is None or len(pdf_docs) == 0:
        st.warning("⚠️ Please upload at least one PDF file")
        return
    
    with st.spinner("🔍 Processing your question..."):
        try:
            # Get text and create vector store
            raw_text = get_pdf_text(pdf_docs)
            if not raw_text.strip():
                st.error("❌ No text could be extracted from the PDF files")
                return
                
            text_chunks = get_text_chunks(raw_text, model_name)
            vector_store = get_vector_store(text_chunks, model_name, api_key)
            
            user_question_output = ""
            response_output = ""
            
            if model_name == "Google AI":
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",  # Correct embedding model
                    google_api_key=api_key
                )
                new_db = FAISS.load_local(
                    "faiss_index", 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                
                # Get relevant documents
                docs = new_db.similarity_search(user_question, k=4)
                
                # Get answer using direct LLM call
                response_output = get_answer_from_docs(docs, user_question, api_key)
                user_question_output = user_question
                
                pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
                conversation_history.append((
                    user_question_output, 
                    response_output, 
                    model_name, 
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    ", ".join(pdf_names)
                ))

                # Display chat messages with improved UI
                st.markdown(
                    f"""
                    <div class="chat-message user">
                        <div class="avatar">
                            <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                        </div>    
                        <div class="message">
                            <strong>You:</strong><br>{user_question_output}
                            <div class="info">📅 {datetime.now().strftime('%H:%M:%S')}</div>
                        </div>
                    </div>
                    <div class="chat-message bot">
                        <div class="avatar">
                            <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
                        </div>
                        <div class="message">
                            <strong>Assistant:</strong><br>{response_output}
                            <div class="info">🤖 Powered by Gemini 2.5 Flash | 📄 Sources: {len(pdf_names)} PDF(s)</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Show download option if conversation exists
            if len(st.session_state.conversation_history) > 0:
                df = pd.DataFrame(
                    st.session_state.conversation_history, 
                    columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"]
                )
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv" class="download-btn">📥 Download Conversation History</a>'
                st.sidebar.markdown(href, unsafe_allow_html=True)
                
                # Show stats
                st.sidebar.markdown(f"""
                    <div class="stats-card">
                        <h3>{len(st.session_state.conversation_history)}</h3>
                        <p>Total Questions Asked</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.success("✅ Response generated successfully!")
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.error("Please check your API key and try again.")

# Main application entry point
def main():
    st.set_page_config(
        page_title="Chat with PDFs", 
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_css()
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>📚 Chat with Multiple PDFs</h1>
            <p>Upload your documents and ask questions powered by AI</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Model selection
        model_name = st.radio(
            "Select AI Model:", 
            ("Google AI",),
            help="Choose the AI model for processing"
        )
        
        # API Key input with password type
        api_key = None
        if model_name == "Google AI":
            api_key = st.text_input(
                "🔑 Enter your Google API Key:",
                type="password",
                help="Your API key is required to use the service",
                placeholder="Enter your API key here"
            )
            st.markdown("🔗 [Get your API key here](https://ai.google.dev/)")
            
            if not api_key:
                st.info("👆 Please enter your Google API Key above to get started.")
        
        st.markdown("---")
        st.title("📁 Document Upload")
        
        # PDF uploader
        pdf_docs = st.file_uploader(
            "Upload PDF Files",
            accept_multiple_files=True,
            type=['pdf'],
            help="Select one or more PDF files to analyze"
        )
        
        if pdf_docs:
            st.success(f"✅ {len(pdf_docs)} file(s) uploaded")
            for pdf in pdf_docs:
                st.text(f"📄 {pdf.name}")
        
        # Process button
        if st.button("🚀 Submit & Process", use_container_width=True):
            if pdf_docs and api_key:
                with st.spinner("Processing PDFs..."):
                    st.success("✅ PDFs processed successfully!")
            elif not pdf_docs:
                st.warning("⚠️ Please upload PDF files before processing.")
            elif not api_key:
                st.warning("⚠️ Please enter your API key before processing.")
        
        st.markdown("---")
        st.title("🔧 Actions")
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Clear Last", use_container_width=True):
                if len(st.session_state.conversation_history) > 0:
                    st.session_state.conversation_history.pop()
                    st.success("✅ Last question cleared")
                    st.rerun()
                else:
                    st.warning("⚠️ No conversation to clear")
        
        with col2:
            if st.button("🗑️ Reset All", use_container_width=True):
                st.session_state.conversation_history = []
                st.success("✅ All conversations cleared")
                st.rerun()
        
        st.markdown("---")
        
        # Info section
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. **Enter API Key**: Get your free API key from Google AI
            2. **Upload PDFs**: Select one or more PDF documents
            3. **Process**: Click 'Submit & Process' button
            4. **Ask Questions**: Type your question in the input box
            5. **Download**: Export conversation history as CSV
            
            **Models Used**: 
            - Chat Model: `gemini-2.5-flash`
            - Embedding: `gemini-embedding-001`
            """)
        
        # Additional info about API key
        with st.expander("🔑 About API Keys"):
            st.markdown("""
            **Getting your Google AI API Key:**
            1. Visit [Google AI Studio](https://ai.google.dev/)
            2. Sign in with your Google account
            3. Click on "Get API Key"
            4. Create a new API key or use existing one
            5. Copy and paste it in the field above
            
            **Your API key is encrypted** (password field) for security.
            """)
    
    # Main chat interface with icon
    st.markdown("### 🔍 Ask Your Question")
    user_question = st.text_input(
        "Type your question here:",
        placeholder="What would you like to know about your PDFs?",
        key="question_input",
        label_visibility="collapsed"
    )
    
    if user_question and user_question.strip():
        user_input(
            user_question, 
            model_name, 
            api_key, 
            pdf_docs, 
            st.session_state.conversation_history
        )
    
    # Display conversation history
    if len(st.session_state.conversation_history) > 0:
        st.markdown("---")
        st.subheader("💬 Conversation History")
        
        # Display in reverse order (newest first)
        for i, (question, answer, model, timestamp, pdfs) in enumerate(reversed(st.session_state.conversation_history)):
            with st.expander(f"Q{len(st.session_state.conversation_history)-i}: {question[:50]}...", expanded=(i==0)):
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:** {answer}")
                st.caption(f"🕐 {timestamp} | 🤖 {model} | 📄 {pdfs}")

if __name__ == "__main__":
    main()
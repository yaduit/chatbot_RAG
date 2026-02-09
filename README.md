# Chat with PDF – RAG Application

A simple **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and ask questions based on their content.  
The system retrieves relevant document chunks using **FAISS** and generates grounded answers using **Google Gemini**.

---

## Tech Stack
- Python
- Streamlit
- LangChain
- FAISS
- Google Gemini API

---

## How the App Works

1. **Upload PDF**  
   The user uploads a PDF document through the Streamlit interface.

2. **Text Extraction & Chunking**  
   The PDF text is extracted and split into smaller, meaningful chunks.

3. **Embedding & Storage**  
   Each chunk is converted into vector embeddings and stored in a FAISS vector database.

4. **Question Answering**  
   When the user asks a question:
   - Relevant chunks are retrieved using semantic similarity search.
   - The retrieved context is passed to the Gemini LLM.
   - The model generates an answer grounded only in the document content.

This approach reduces hallucinations and ensures responses are based on the uploaded PDF.

---

## How to Run the Project
```bash

### 1. Clone the repository
git clone https://github.com/your-username/chat-with-pdf-rag.git
cd chat-with-pdf-rag

### 2. Create and activate a virtual environment
python -m venv myenv
myenv\Scripts\activate        # Windows
source myenv/bin/activate     # macOS/Linux

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run the application
streamlit run app.py

### 5. Set up the Gemini API key


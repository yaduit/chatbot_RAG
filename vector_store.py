import os
import hashlib
import pickle
from typing import List, Tuple
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

#configuration

CACHE_DIR = Path(".vector_cache")  # Directory to store cached FAISS indexes
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, lightweight model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# HELPER FUNCTIONS

def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(exist_ok=True)


def _get_pdf_hash(pdf_file) -> str:
    """
    Generate unique hash for a PDF file.
    
    Uses: filename + file size + first 1MB content
    This ensures we detect when PDF actually changes.
    
    Args:
        pdf_file: Streamlit UploadedFile object
        
    Returns:
        str: Unique hash for this PDF
    """
    # Reset file pointer to beginning
    pdf_file.seek(0)
    
    # Read first 1MB for content hashing (efficient for large files)
    content_sample = pdf_file.read(1024 * 1024)
    pdf_file.seek(0)  # Reset again for later use
    
    # Combine filename, size, and content sample
    hash_input = f"{pdf_file.name}_{pdf_file.size}_{content_sample}".encode()
    return hashlib.md5(hash_input).hexdigest()


def _get_cache_path(pdf_hash: str) -> Tuple[Path, Path]:
    """
    Get file paths for cached FAISS index and metadata.
    
    Args:
        pdf_hash: Unique identifier for PDF
        
    Returns:
        Tuple of (faiss_path, metadata_path)
    """
    _ensure_cache_dir()
    faiss_path = CACHE_DIR / f"{pdf_hash}_faiss"
    metadata_path = CACHE_DIR / f"{pdf_hash}_metadata.pkl"
    return faiss_path, metadata_path


def _cache_exists(pdf_hash: str) -> bool:
    """
    Check if valid cache exists for this PDF.
    
    Args:
        pdf_hash: Unique identifier for PDF
        
    Returns:
        bool: True if cache exists and is valid
    """
    faiss_path, metadata_path = _get_cache_path(pdf_hash)
    
    # Check if both FAISS index and metadata exist
    faiss_index_exists = faiss_path.exists() and (faiss_path / "index.faiss").exists()
    metadata_exists = metadata_path.exists()
    
    return faiss_index_exists and metadata_exists


# CORE FUNCTIONS

def get_embeddings_model():
    """
    Get the embeddings model (cached by Streamlit).
    
    Uses sentence-transformers (local, no API needed).
    Model is loaded once and reused across all operations.
    
    Returns:
        HuggingFaceEmbeddings: Embeddings model instance
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
        encode_kwargs={'normalize_embeddings': True}
    )


def create_text_chunks(text: str) -> List[str]:
    """
    Split text into chunks for embedding.
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    return chunks


def get_or_create_vector_store(pdf_file, raw_text: str, embeddings_model) -> FAISS:
    """
    Get cached vector store or create new one.
    
    This is the CORE caching function. It:
    1. Generates unique hash for PDF
    2. Checks if cached FAISS index exists
    3. If yes: Load from disk (FAST)
    4. If no: Generate embeddings and save to disk
    
    Args:
        pdf_file: Streamlit UploadedFile object
        raw_text: Extracted text from PDF
        embeddings_model: HuggingFace embeddings model
        
    Returns:
        FAISS: Vector store (either cached or newly created)
    """
    # Generate unique identifier for this PDF
    pdf_hash = _get_pdf_hash(pdf_file)
    
    # Check if we have a cached version
    if _cache_exists(pdf_hash):
        print(f"✅ Loading cached FAISS index for: {pdf_file.name}")
        return _load_from_cache(pdf_hash, embeddings_model)
    
    # No cache found - create new embeddings
    print(f"🔄 Creating new FAISS index for: {pdf_file.name}")
    vector_store = _create_and_cache_vector_store(
        pdf_hash, 
        raw_text, 
        embeddings_model,
        pdf_file.name
    )
    
    return vector_store


def _load_from_cache(pdf_hash: str, embeddings_model) -> FAISS:
    """
    Load FAISS index from disk cache.
    
    Args:
        pdf_hash: Unique identifier for PDF
        embeddings_model: Embeddings model (needed for FAISS)
        
    Returns:
        FAISS: Loaded vector store
    """
    faiss_path, metadata_path = _get_cache_path(pdf_hash)
    
    # Load metadata (for debugging/logging)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"   📊 Chunks: {metadata['num_chunks']}")
    print(f"   📅 Cached: {metadata['created_at']}")
    
    # Load FAISS index from disk
    # allow_dangerous_deserialization=True is safe here because we control the cache
    vector_store = FAISS.load_local(
        str(faiss_path),
        embeddings_model,
        allow_dangerous_deserialization=True
    )
    
    return vector_store


def _create_and_cache_vector_store(
    pdf_hash: str, 
    raw_text: str, 
    embeddings_model,
    pdf_name: str
) -> FAISS:
    """
    Create new FAISS index and save to disk.
    
    This is where embeddings are actually generated.
    Called ONLY when cache doesn't exist.
    
    Args:
        pdf_hash: Unique identifier for PDF
        raw_text: Extracted text from PDF
        embeddings_model: Embeddings model
        pdf_name: Name of PDF (for metadata)
        
    Returns:
        FAISS: Newly created vector store
    """
    from datetime import datetime
    
    # Split text into chunks
    chunks = create_text_chunks(raw_text)
    print(f"   📊 Created {len(chunks)} chunks")
    
    # Generate embeddings and create FAISS index
    # This is the EXPENSIVE operation we want to cache
    print(f"   🔄 Generating embeddings...")
    vector_store = FAISS.from_texts(chunks, embeddings_model)
    
    # Save to disk for future use
    faiss_path, metadata_path = _get_cache_path(pdf_hash)
    
    # Save FAISS index
    vector_store.save_local(str(faiss_path))
    
    # Save metadata
    metadata = {
        'pdf_name': pdf_name,
        'pdf_hash': pdf_hash,
        'num_chunks': len(chunks),
        'created_at': datetime.now().isoformat(),
        'embedding_model': EMBEDDING_MODEL
    }
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"   ✅ Cached to disk")
    
    return vector_store


def similarity_search(vector_store: FAISS, query: str, k: int = 4) -> List[Document]:
    """
    Search for similar documents in vector store.
    
    Args:
        vector_store: FAISS vector store
        query: User question
        k: Number of results to return
        
    Returns:
        List of relevant document chunks
    """
    return vector_store.similarity_search(query, k=k)


def clear_cache():
    """
    Clear all cached FAISS indexes.
    
    Useful for:
    - Freeing disk space
    - Forcing regeneration
    - Debugging
    """
    import shutil
    
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        _ensure_cache_dir()
        print("🗑️ Cache cleared")
    else:
        print("ℹ️ No cache to clear")


def get_cache_info() -> dict:
    """
    Get information about cached indexes.
    
    Returns:
        dict: Cache statistics
    """
    if not CACHE_DIR.exists():
        return {
            'num_cached_pdfs': 0,
            'total_size_mb': 0,
            'cached_files': []
        }
    
    cached_files = []
    total_size = 0
    
    # Find all metadata files
    for metadata_file in CACHE_DIR.glob("*_metadata.pkl"):
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
            cached_files.append(metadata)
        
        # Calculate size
        pdf_hash = metadata['pdf_hash']
        faiss_path, _ = _get_cache_path(pdf_hash)
        if faiss_path.exists():
            total_size += sum(f.stat().st_size for f in faiss_path.rglob('*') if f.is_file())
    
    return {
        'num_cached_pdfs': len(cached_files),
        'total_size_mb': round(total_size / (1024 * 1024), 2),
        'cached_files': cached_files
    }
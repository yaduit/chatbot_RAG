# vector_store.py

import hashlib
import pickle
from pathlib import Path
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# =============================================================================
# CONFIGURATION
# =============================================================================

CACHE_DIR = Path(".vector_cache")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(exist_ok=True)


def _get_pdf_hash(pdf_file) -> str:
    """
    Generate a stable hash for a PDF using:
    - filename
    - size
    - first 1MB of content
    """
    pdf_file.seek(0)
    content_sample = pdf_file.read(1024 * 1024)
    pdf_file.seek(0)

    hash_input = f"{pdf_file.name}_{pdf_file.size}_{content_sample}".encode()
    return hashlib.md5(hash_input).hexdigest()


def _get_cache_paths(pdf_hash: str) -> Tuple[Path, Path]:
    _ensure_cache_dir()
    faiss_path = CACHE_DIR / f"{pdf_hash}_faiss"
    metadata_path = CACHE_DIR / f"{pdf_hash}_meta.pkl"
    return faiss_path, metadata_path


def _cache_exists(pdf_hash: str) -> bool:
    faiss_path, meta_path = _get_cache_paths(pdf_hash)
    return (
        faiss_path.exists()
        and (faiss_path / "index.faiss").exists()
        and meta_path.exists()
    )


# =============================================================================
# PUBLIC API
# =============================================================================

def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Return embeddings model.
    Streamlit caching must be handled in app.py.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_or_create_vector_store(pdf_file, raw_text: str, embeddings) -> FAISS:
    """
    Load FAISS index from disk if available,
    otherwise create and persist it.
    """
    pdf_hash = _get_pdf_hash(pdf_file)

    if _cache_exists(pdf_hash):
        return _load_from_cache(pdf_hash, embeddings)

    return _create_and_cache_vector_store(
        pdf_hash=pdf_hash,
        raw_text=raw_text,
        embeddings=embeddings,
        pdf_name=pdf_file.name,
    )


def similarity_search(
    vector_store: FAISS,
    query: str,
    k: int = 4,
) -> List[Document]:
    return vector_store.similarity_search(query, k=k)


def clear_cache() -> None:
    import shutil

    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(exist_ok=True)


def get_cache_info() -> dict:
    if not CACHE_DIR.exists():
        return {"count": 0, "size_mb": 0}

    total_size = sum(
        f.stat().st_size
        for f in CACHE_DIR.rglob("*")
        if f.is_file()
    )

    return {
        "count": len(list(CACHE_DIR.glob("*_meta.pkl"))),
        "size_mb": round(total_size / (1024 * 1024), 2),
    }


# =============================================================================
# INTERNAL IMPLEMENTATION
# =============================================================================

def _load_from_cache(pdf_hash: str, embeddings) -> FAISS:
    faiss_path, meta_path = _get_cache_paths(pdf_hash)

    with open(meta_path, "rb") as f:
        _ = pickle.load(f)  # metadata (optional use)

    return FAISS.load_local(
        str(faiss_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _create_and_cache_vector_store(
    pdf_hash: str,
    raw_text: str,
    embeddings,
    pdf_name: str,
) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_text(raw_text)

    vector_store = FAISS.from_texts(chunks, embeddings)

    faiss_path, meta_path = _get_cache_paths(pdf_hash)
    vector_store.save_local(str(faiss_path))

    metadata = {
        "pdf_name": pdf_name,
        "pdf_hash": pdf_hash,
        "num_chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
    }

    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    return vector_store


__all__ = [
    "get_embeddings_model",
    "get_or_create_vector_store",
    "similarity_search",
    "clear_cache",
    "get_cache_info",
]

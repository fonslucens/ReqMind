"""Loading component: embed the chunks and store them in ChromaDB."""
import hashlib

from langchain_community.vectorstores import Chroma

from config import CHROMA_DIR, COLLECTION_NAME
from ingestion import load_documents
from transformation import get_embeddings, split_documents


def create_chunk_ids(chunks):
    """Create a unique SHA-256 ID for each chunk (re-running does not add duplicates)."""
    return [
        hashlib.sha256(f"{i}:{chunk.page_content}".encode("utf-8")).hexdigest()
        for i, chunk in enumerate(chunks)
    ]


def build_vector_store():
    """Run the loading pipeline: PDF -> chunks -> embeddings -> Chroma."""
    documents = load_documents()
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    chunk_ids = create_chunk_ids(chunks)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=chunk_ids,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print("Vector store ready for retrieval")
    return vector_store

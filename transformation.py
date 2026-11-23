"""Data transformation component: chunking and embeddings."""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL


def split_documents(documents):
    """Split pages into chunks of 1000 characters with a 100-character overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")
    print("Preview of the first chunk (500 characters):")
    print(chunks[0].page_content[:500])
    print("-" * 60)
    return chunks


def get_embeddings():
    """Create the local Hugging Face embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

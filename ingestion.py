"""Ingestion component: load every PDF in data/ as LangChain Documents."""
from langchain_community.document_loaders import PyPDFLoader

from config import DATA_DIR


def load_documents():
    """Return one Document per PDF page, with source filename and page in metadata."""
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in the data/ folder.")

    documents = []
    for pdf_file in pdf_files:
        pages = PyPDFLoader(str(pdf_file)).load()
        for page in pages:
            page.metadata["source"] = pdf_file.name
        documents.extend(pages)
        print(f"Loaded {pdf_file.name}: {len(pages)} pages")

    return documents

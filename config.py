"""Configuration: loads the Groq API key and model name from the .env file."""
import os
from pathlib import Path

# Protobuf compatibility workaround: must be set before LangChain/Chroma are imported.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

DATA_DIR = PROJECT_DIR / "data"
CHROMA_DIR = str(PROJECT_DIR / "chroma_db")

COLLECTION_NAME = "reqmind"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 3

"""ReqMind: ask questions about a requirements PDF from the terminal.

Client (frontend) component: an interactive chatbot in the terminal.
Run:  python main.py
"""
import config
from loading import build_vector_store
from rag_service import create_rag_chain


def show_answer(result):
    """Print the generated answer and the retrieved source chunks."""
    print("\n--- Answer ---")
    print(result["answer"])

    print("\n--- Source Chunks ---")
    for i, doc in enumerate(result["context"], start=1):
        page = doc.metadata.get("page", 0) + 1
        print(f"Source {i}: {doc.metadata['source']}, page {page}")
        print(doc.page_content[:200].replace("\n", " ") + "...")
        print("-" * 60)


def chat_loop(rag_chain):
    """Ask questions until the user types exit."""
    print("\nReqMind is ready. Ask a question about the requirements.")
    print("Type 'exit' to quit.")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if not question:
            continue
        try:
            result = rag_chain.invoke({"input": question})
            show_answer(result)
        except Exception as error:
            print(f"Could not get an answer: {error}")


def main():
    if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your_key_here":
        raise RuntimeError("GROQ_API_KEY was not found. Check your .env file.")
    if not config.GROQ_MODEL:
        raise RuntimeError("GROQ_MODEL was not found. Check your .env file.")

    print("Loading data...")
    vector_store = build_vector_store()
    rag_chain = create_rag_chain(vector_store)
    chat_loop(rag_chain)


if __name__ == "__main__":
    main()

"""RAG service component: retriever + Groq LLM + prompt chain."""
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MODEL, TOP_K


def create_rag_chain(vector_store):
    """Connect the retriever (k=3) and the Groq model into one retrieval chain."""
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    # ChatGroq is the adapter between LangChain's chat-model interface and the Groq API.
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.1,
        max_retries=2,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are ReqMind, an assistant that helps engineers find "
                "software requirements. Answer the question using only the "
                "supplied context. Include the requirement IDs (for example "
                "NFR-01) when they appear in the context. If the context does "
                "not contain the answer, say that the requirements document "
                "does not contain this information.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    print("RAG chain successfully created")
    return rag_chain

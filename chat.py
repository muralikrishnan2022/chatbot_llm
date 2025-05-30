import os
import signal
import sys
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from models import Models

load_dotenv()

# Initialize the models
try:
    models = Models()
    embeddings = models.embeddings_ollama
    llm = models.model_ollama
except Exception as e:
    print(f"Error initializing models: {str(e)}")
    sys.exit(1)

# Initialize the vector store
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./data/chroma_langchain_db")
try:
    vector_store = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
except Exception as e:
    print(f"Error initializing vector store: {str(e)}")
    sys.exit(1)

# Define the chat prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and factual assistant. Ask the user complete information before making any conclusion. Don't make decisions based on partial information provided by the user. You must only respond based on the facts explicitly provided by the user. If any part of the user's question is ambiguous or lacks critical details, ask direct and specific clarifying questions before providing an answer. By default, assume users are government employees unless otherwise stated."),
    ("human", "{context}\n\nQuestion: {input}")
])

# Define the retrieval chain
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 3))
try:
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
except Exception as e:
    print(f"Error setting up retrieval chain: {str(e)}")
    sys.exit(1)

def signal_handler(sig, frame):
    """Handle termination signals for graceful exit."""
    print("\nReceived termination signal. Exiting gracefully...")
    sys.exit(0)

def main():
    """Main loop for user interaction."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while True:
        query = input("User (or type 'q', 'quit', or 'exit' to end): ").strip()
        if query.lower() in ['q', 'quit', 'exit']:
            break
        if not query:
            print("Assistant: Please enter a valid question.\n")
            continue

        try:
            result = retrieval_chain.invoke({"input": query})
            if not result.get("context"):
                print("Assistant: No relevant information found in the documents.\n")
            else:
                print("Assistant:", result["answer"], "\n")
        except Exception as e:
            print(f"Assistant: Error processing query: {str(e)}\n")

if __name__ == "__main__":
    main()
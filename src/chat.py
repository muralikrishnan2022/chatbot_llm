import sys
import unicodedata
import os
import shutil
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from transformers import MarianTokenizer, MarianMTModel
import logging
from config import CONFIG, initialize_models, initialize_vector_stores, setup_signal_handlers
from uuid import uuid4

logger = logging.getLogger(__name__)

# Ensure terminal encoding is UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def store_chat_history(vector_store, query, answer):
    """Store chat history in the user's ChromaDB collection."""
    try:
        chat_text = f"User: {query}\nAssistant: {answer}"
        document = Document(page_content=chat_text, metadata={"type": "chat_history"})
        vector_store.add_documents(documents=[document], ids=[str(uuid4())])
        logger.info("Stored chat history")
    except Exception as e:
        logger.error(f"Error storing chat history: {str(e)}")

def view_chat_history(vector_store):
    """Retrieve and display chat history from the user's ChromaDB collection."""
    try:
        # Query documents with metadata type="chat_history"
        results = vector_store.get(where={"type": "chat_history"})
        if not results.get("documents"):
            print("No chat history found.\n")
            print("कोई चैट इतिहास नहीं मिला।\n")
            return

        print("\nChat History:")
        print("-------------")
        for doc in results["documents"]:
            print(doc.strip())
            print("-------------")
        logger.info("Displayed chat history")
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        print(f"Error retrieving chat history: {str(e)}\n")

def main():
    """Main loop for user interaction."""
    setup_signal_handlers()
    models = initialize_models()

    # Prompt for user ID
    user_id = input("Enter your user ID: ").strip()
    if not user_id:
        logger.error("User ID is required")
        print("User ID is required.")
        sys.exit(1)

    # Sanitize user_id to prevent path traversal
    user_id = user_id.replace('/', '_').replace('\\', '_').replace('..', '_')

    # Initialize both vector stores
    default_vector_store, user_vector_store = initialize_vector_stores(models.embeddings_ollama, user_id)

    # Create uploads directory for the user
    upload_dir = os.path.join(CONFIG["DATA_FOLDER"], f"user_{user_id}")
    os.makedirs(upload_dir, exist_ok=True)

    # Initialize translation model
    try:
        save_directory = "en-hi-model"
        tokenizer = MarianTokenizer.from_pretrained(save_directory)
        model = MarianMTModel.from_pretrained(save_directory, use_safetensors=True)
        logger.info("Initialized translation model")
    except Exception as e:
        logger.error(f"Error initializing translation model: {str(e)}")
        sys.exit(1)

    # Define the chat prompt for book queries
    book_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question based only on the provided context from the book."),
        ("human", "{context}\n\nQuestion: {input}")
    ])

    # Define the chat prompt for user document queries
    user_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question based only on the provided context from user-uploaded documents and chat history."),
        ("human", "{context}\n\nQuestion: {input}")
    ])

    # Set up retrieval chains
    try:
        # Book retrieval chain
        book_retriever = default_vector_store.as_retriever(search_kwargs={"k": CONFIG["RETRIEVER_K"]})
        book_combine_docs_chain = create_stuff_documents_chain(models.model_ollama, book_prompt)
        book_retrieval_chain = create_retrieval_chain(book_retriever, book_combine_docs_chain)

        # User retrieval chain (only if user_vector_store exists)
        user_retrieval_chain = None
        if user_vector_store:
            user_retriever = user_vector_store.as_retriever(search_kwargs={"k": CONFIG["RETRIEVER_K"]})
            user_combine_docs_chain = create_stuff_documents_chain(models.model_ollama, user_prompt)
            user_retrieval_chain = create_retrieval_chain(user_retriever, user_combine_docs_chain)

        logger.info("Initialized retrieval chains")
    except Exception as e:
        logger.error(f"Error setting up retrieval chains: {str(e)}")
        sys.exit(1)

    while True:
        print("\nOptions: (1) Upload file, (2) Ask a question about the book, (3) Ask a question about uploaded documents, (4) View chat history, (5) Exit")
        choice = input("Choose an option: ").strip()

        if choice == "1":
            file_path = input("Enter the path to the file (PDF or image): ").strip()
            if os.path.exists(file_path):
                try:
                    # Copy file to user-specific upload directory using shutil
                    destination = os.path.join(upload_dir, os.path.basename(file_path))
                    shutil.copy(file_path, destination)
                    logger.info(f"Copied file from {file_path} to {destination}")
                    # Ingest the file into user-specific vector store
                    from ingest import ingest_file
                    ingest_file(destination, user_vector_store)
                except Exception as e:
                    logger.error(f"Error copying or ingesting file {file_path}: {str(e)}")
                    print(f"Error uploading file: {str(e)}")
            else:
                print("File does not exist.")
                logger.info(f"Invalid file path: {file_path}")

        elif choice == "2":
            query = input("Enter your question about the book: ").strip()
            if not query:
                logger.info("Empty query received")
                print("Assistant: Please enter a valid question.\n")
                print("सहायक: कृपया एक मान्य प्रश्न दर्ज करें।\n")
                continue

            try:
                result = book_retrieval_chain.invoke({"input": query})
                if not result.get("context"):
                    logger.info("No relevant information found in the book")
                    print("Assistant: No relevant information found in the book.\n")
                    print("सहायक: पुस्तक में कोई प्रासंगिक जानकारी नहीं मिली।\n")
                else:
                    logger.info(f"Retrieved {len(result['context'])} documents from book collection")
                    answer = result["answer"]
                    print("Assistant:", answer, "\n")
                    # Store chat history in user-specific vector store
                    if user_vector_store:
                        store_chat_history(user_vector_store, query, answer)
                    # Translate answer to Hindi
                    inputs = tokenizer(str(answer), return_tensors="pt", padding=True)
                    translated = model.generate(**inputs)
                    decoded_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
                    for text in decoded_texts:
                        normalized_text = unicodedata.normalize('NFC', text)
                        print(f"सहायक: {normalized_text}")
            except Exception as e:
                logger.error(f"Error processing book query: {str(e)}")
                print(f"Assistant: Error processing query: {str(e)}\n")

        elif choice == "3":
            if not user_vector_store:
                print("Assistant: No user-specific documents available. Please upload a file first.\n")
                print("सहायक: कोई उपयोगकर्ता-विशिष्ट दस्तावेज़ उपलब्ध नहीं हैं। कृपया पहले एक फ़ाइल अपलोड करें।\n")
                continue

            query = input("Enter your question about uploaded documents: ").strip()
            if not query:
                logger.info("Empty query received")
                print("Assistant: Please enter a valid question.\n")
                print("सहायक: कृपया एक मान्य प्रश्न दर्ज करें।\n")
                continue

            try:
                result = user_retrieval_chain.invoke({"input": query})
                if not result.get("context"):
                    logger.info("No relevant information found in uploaded documents")
                    print("Assistant: No relevant information found in uploaded documents.\n")
                    print("सहायक: अपलोड किए गए दस्तावेज़ों में कोई प्रासंगिक जानकारी नहीं मिली।\n")
                else:
                    logger.info(f"Retrieved {len(result['context'])} documents from user collection")
                    answer = result["answer"]
                    print("Assistant:", answer, "\n")
                    # Store chat history in user-specific vector store
                    store_chat_history(user_vector_store, query, answer)
                    # Translate answer to Hindi
                    inputs = tokenizer(str(answer), return_tensors="pt", padding=True)
                    translated = model.generate(**inputs)
                    decoded_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
                    for text in decoded_texts:
                        normalized_text = unicodedata.normalize('NFC', text)
                        print(f"सहायक: {normalized_text}")
            except Exception as e:
                logger.error(f"Error processing user document query: {str(e)}")
                print(f"Assistant: Error processing query: {str(e)}\n")

        elif choice == "4":
            if not user_vector_store:
                print("Assistant: No user-specific chat history available.\n")
                print("सहायक: कोई उपयोगकर्ता-विशिष्ट चैट इतिहास उपलब्ध नहीं है।\n")
                continue
            view_chat_history(user_vector_store)

        elif choice == "5":
            logger.info("Exiting chatbot")
            print("Exiting chatbot.")
            break

        else:
            print("Invalid option. Please choose 1, 2, 3, 4, or 5.")

if __name__ == "__main__":
    main()
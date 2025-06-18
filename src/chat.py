import sys
import unicodedata
import os
import shutil
import sqlite3
import bcrypt
import getpass
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
import logging
from config import CONFIG, initialize_models, initialize_vector_stores, setup_signal_handlers
from uuid import uuid4

logger = logging.getLogger(__name__)

# Ensure terminal encoding is UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Database setup
DB_PATH = os.path.join(CONFIG["DATA_FOLDER"], "users.db")

def initialize_database():
    """Initialize SQLite database for user storage."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL
                )
            """)
            conn.commit()
            logger.info("Initialized user database")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        sys.exit(1)

def register_user(username, password):
    """Register a new user with hashed password."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                logger.warning(f"Registration attempt for existing username: {username}")
                return False, "Username already exists."

            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash)
            )
            conn.commit()
            logger.info(f"User {username} registered successfully")
            return True, "Registration successful."
    except Exception as e:
        logger.error(f"Error registering user {username}: {str(e)}")
        return False, f"Error registering user: {str(e)}"

def authenticate_user():
    """Authenticate user with username and password, return username and session_id."""
    max_attempts = 3
    print("Enter 'register' as username to create a new account.")
    
    for attempt in range(max_attempts):
        username = input("Enter username: ").strip()
        
        if username.lower() == 'register':
            new_username = input("Enter new username: ").strip()
            if not new_username or '/' in new_username or '\\' in new_username or '..' in new_username:
                print("Invalid username. Avoid special characters like /, \\, or ..")
                continue
            password = getpass.getpass("Enter new password: ").strip()
            if not password:
                print("Password cannot be empty.")
                continue
            success, message = register_user(new_username, password)
            print(message)
            return new_username, str(uuid4()) if success else (None, None)
        
        password = getpass.getpass("Enter password: ").strip()
        
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
                result = cursor.fetchone()
                
                if result and bcrypt.checkpw(password.encode('utf-8'), result[0]):
                    session_id = str(uuid4())
                    logger.info(f"User {username} authenticated with session_id: {session_id}")
                    return username, session_id
                else:
                    remaining = max_attempts - attempt - 1
                    if remaining > 0:
                        print(f"Invalid credentials. {remaining} attempts remaining.")
                    else:
                        print("Too many failed attempts. Exiting.")
                        logger.error("Authentication failed: too many attempts")
                        sys.exit(1)
        except Exception as e:
            logger.error(f"Error during authentication: {str(e)}")
            print(f"Error during authentication: {str(e)}")
            remaining = max_attempts - attempt - 1
            if remaining == 0:
                sys.exit(1)
    return None, None

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

    # Initialize database
    initialize_database()

    # Authenticate user
    username, session_id = authenticate_user()
    if not username or not session_id:
        sys.exit(1)

    # Use username as user_id
    user_id = username

    # Sanitize user_id to prevent path traversal
    user_id = user_id.replace('/', '_').replace('\\', '_').replace('..', '_')

    # Initialize vector stores
    default_vector_store, user_vector_store, session_vector_store = initialize_vector_stores(
        models.embeddings_ollama, user_id, session_id
    )

    # Create uploads directory for the user
    upload_dir = os.path.join(CONFIG["DATA_FOLDER"], f"user_{user_id}")
    os.makedirs(upload_dir, exist_ok=True)

    # Define the chat prompts for book queries (English and Hindi)
    book_prompt_english = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question based only on the provided context from the book."),
        ("human", "{context}\n\nQuestion: {input}")
    ])
    book_prompt_hindi = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question based only on the provided context from the book. Please provide in Hindi."),
        ("human", "{context}\n\nQuestion: {input}")
    ])

    # Define the chat prompts for user document queries (English and Hindi)
    user_prompt_english = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question based only on the provided context from user-uploaded documents."),
        ("human", "{context}\n\nQuestion: {input}")
    ])
    user_prompt_hindi = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question based only on the provided context from user-uploaded documents. Please provide in Hindi."),
        ("human", "{context}\n\nQuestion: {input}")
    ])

    # Language toggle flag
    use_hindi = False

    # Set up retrieval chains (initially in English)
    try:
        book_retriever = default_vector_store.as_retriever(search_kwargs={"k": CONFIG["RETRIEVER_K"]})
        book_combine_docs_chain = create_stuff_documents_chain(models.model_ollama, book_prompt_english)
        book_retrieval_chain = create_retrieval_chain(book_retriever, book_combine_docs_chain)

        session_retrieval_chain = None
        if session_vector_store:
            session_retriever = session_vector_store.as_retriever(search_kwargs={"k": CONFIG["RETRIEVER_K"]})
            session_combine_docs_chain = create_stuff_documents_chain(models.model_ollama, user_prompt_english)
            session_retrieval_chain = create_retrieval_chain(session_retriever, session_combine_docs_chain)

        logger.info("Initialized retrieval chains")
    except Exception as e:
        logger.error(f"Error setting up retrieval chains: {str(e)}")
        sys.exit(1)

    print(f"\nWelcome, {username}!")

    # Track whether documents have been uploaded in this session
    documents_uploaded = False

    while True:
        print("\nOptions: (1) Upload file, (2) Ask a question about the book, (3) Ask a question about uploaded documents, (4) View chat history, (5) Exit, (6) Toggle language (English/Hindi)")
        choice = input("Choose an option: ").strip()

        if choice == "1":
            file_path = input("Enter the path to the file (PDF or image): ").strip()
            if os.path.exists(file_path):
                try:
                    destination = os.path.join(upload_dir, os.path.basename(file_path))
                    shutil.copy(file_path, destination)
                    logger.info(f"Copied file from {file_path} to {destination}")
                    from ingest import ingest_file
                    ingest_file(destination, session_vector_store)
                    documents_uploaded = True
                    print("File uploaded successfully." if not use_hindi else "फ़ाइल सफलतापूर्वक अपलोड की गई।")
                except Exception as e:
                    logger.error(f"Error copying or ingesting file {file_path}: {str(e)}")
                    print(f"Error uploading file: {str(e)}" if not use_hindi else f"फ़ाइल अपलोड करने में त्रुटि: {str(e)}")
            else:
                print("File does not exist." if not use_hindi else "फ़ाइल मौजूद नहीं है।")
                logger.info(f"Invalid file path: {file_path}")

        elif choice == "2":
            query = input("Enter your question about the book: ").strip()
            if not query:
                logger.info("Empty query received")
                print("Assistant: Please enter a valid question." if not use_hindi else "सहायक: कृपया एक मान्य प्रश्न दर्ज करें।")
                continue

            try:
                result = book_retrieval_chain.invoke({"input": query})
                if not result.get("context"):
                    logger.info("No relevant information found in the book")
                    print("Assistant: No relevant information found in the book." if not use_hindi else "सहायक: पुस्तक में कोई प्रासंगिक जानकारी नहीं मिली।")
                else:
                    logger.info(f"Retrieved {len(result['context'])} documents from book collection")
                    answer = result["answer"]
                    print("Assistant:", answer)
                    if user_vector_store:
                        store_chat_history(user_vector_store, query, answer)
            except Exception as e:
                logger.error(f"Error processing book query: {str(e)}")
                print(f"Assistant: Error processing query: {str(e)}" if not use_hindi else f"सहायक: प्रश्न संसाधित करने में त्रुटि: {str(e)}")

        elif choice == "3":
            if not session_vector_store or not documents_uploaded:
                print("Assistant: No documents available. Please upload a file first." if not use_hindi else "सहायक: कोई दस्तावेज़ उपलब्ध नहीं हैं। कृपया पहले एक फ़ाइल अपलोड करें।")
                continue

            query = input("Enter your question about uploaded documents: ").strip()
            if not query:
                logger.info("Empty query received")
                print("Assistant: Please enter a valid question." if not use_hindi else "सहायक: कृपया एक मान्य प्रश्न दर्ज करें।")
                continue

            try:
                result = session_retrieval_chain.invoke({"input": query})
                if not result.get("context"):
                    logger.info("No relevant information found in uploaded documents")
                    print("Assistant: No relevant information found in uploaded documents." if not use_hindi else "सहायक: अपलोड किए गए दस्तावेज़ों में कोई प्रासंगिक जानकारी नहीं मिली।")
                else:
                    logger.info(f"Retrieved {len(result['context'])} documents from session collection")
                    answer = result["answer"]
                    print("Assistant:", answer)
                    if user_vector_store:
                        store_chat_history(user_vector_store, query, answer)
            except Exception as e:
                logger.error(f"Error processing user document query: {str(e)}")
                print(f"Assistant: Error processing query: {str(e)}" if not use_hindi else f"सहायक: प्रश्न संसाधित करने में त्रुटि: {str(e)}")

        elif choice == "4":
            if not user_vector_store:
                print("Assistant: No chat history available." if not use_hindi else "सहायक: कोई चैट इतिहास उपलब्ध नहीं है।")
                continue
            view_chat_history(user_vector_store)

        elif choice == "5":
            logger.info("Exiting chatbot")
            print("Exiting chatbot." if not use_hindi else "चैटबॉट से बाहर निकल रहे हैं।")
            if session_vector_store:
                try:
                    session_vector_store.delete_collection()
                    logger.info(f"Deleted session vector store for session: {session_id}")
                except Exception as e:
                    logger.error(f"Error deleting session vector store: {str(e)}")
            try:
                shutil.rmtree(upload_dir, ignore_errors=True)
                logger.info(f"Deleted user upload directory: {upload_dir}")
            except Exception as e:
                logger.error(f"Error deleting user upload directory: {str(e)}")
            break

        elif choice == "6":
            # Toggle language
            use_hindi = not use_hindi
            # Update retrieval chains based on language
            try:
                book_combine_docs_chain = create_stuff_documents_chain(
                    models.model_ollama, book_prompt_hindi if use_hindi else book_prompt_english
                )
                book_retrieval_chain = create_retrieval_chain(book_retriever, book_combine_docs_chain)

                if session_vector_store:
                    session_combine_docs_chain = create_stuff_documents_chain(
                        models.model_ollama, user_prompt_hindi if use_hindi else user_prompt_english
                    )
                    session_retrieval_chain = create_retrieval_chain(session_retriever, session_combine_docs_chain)

                logger.info(f"Switched to {'Hindi' if use_hindi else 'English'} mode")
                print(f"Switched to {'Hindi' if use_hindi else 'English'} mode." if not use_hindi else f"{'हिंदी' if use_hindi else 'अंग्रेजी'} मोड में स्विच किया गया।")
            except Exception as e:
                logger.error(f"Error switching language: {str(e)}")
                print(f"Error switching language: {str(e)}" if not use_hindi else f"भाषा स्विच करने में त्रुटि: {str(e)}")

        else:
            print("Invalid option. Please choose 1, 2, 3, 4, 5, or 6." if not use_hindi else "अमान्य विकल्प। कृपया 1, 2, 3, 4, 5, या 6 चुनें।")

if __name__ == "__main__":
    main()
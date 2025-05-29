import os
import time
import signal
import sys
from uuid import uuid4
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from models import Models

load_dotenv()

# Initialize the models
try:
    models = Models()
    embeddings = models.embeddings_ollama
except Exception as e:
    print(f"Error initializing models: {str(e)}")
    sys.exit(1)

# Define constants from environment variables with defaults
DATA_FOLDER = os.getenv("DATA_FOLDER", "./pdfs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 10))
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./data/chroma_langchain_db")

# Initialize Chroma vector store
try:
    vector_store = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
except Exception as e:
    print(f"Error initializing vector store: {str(e)}")
    sys.exit(1)

# Track processed files to avoid duplicates
PROCESSED_FILES = set()

def ingest_file(file_path):
    """Ingest a single PDF file into the vector store."""
    # Skip if non-PDF or already processed
    if not file_path.lower().endswith('.pdf'):
        print(f"Skipping non-PDF file: {file_path}")
        return
    if file_path in PROCESSED_FILES:
        print(f"Skipping already processed file: {file_path}")
        return

    try:
        print(f"Starting to ingest file: {file_path}")
        loader = PyMuPDFLoader(file_path)
        loaded_documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n", " ", ""]
        )

        documents = text_splitter.split_documents(loaded_documents)
        uuids = [str(uuid4()) for _ in range(len(documents))]

        print(f"Adding {len(documents)} documents to the vector store")
        vector_store.add_documents(documents=documents, ids=uuids)
        PROCESSED_FILES.add(file_path)
        print(f"Finished ingesting file: {file_path}")

        # Rename file to mark as processed
        base, ext = os.path.splitext(file_path)
        new_file_path = f"{base}_processed{ext}"
        counter = 1
        while os.path.exists(new_file_path):
            new_file_path = f"{base}_processed_{counter}{ext}"
            counter += 1
        os.rename(file_path, new_file_path)
        print(f"Renamed file to: {new_file_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")

def signal_handler(sig, frame):
    """Handle termination signals for graceful exit."""
    print("\nReceived termination signal. Exiting gracefully...")
    sys.exit(0)

def main_loop():
    """Main loop to monitor and ingest PDF files."""
    # Register signal handlers for graceful termination
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while True:
        try:
            for filename in os.listdir(DATA_FOLDER):
                file_path = os.path.join(DATA_FOLDER, filename)
                if not filename.startswith("_"):  # Skip already processed files
                    ingest_file(file_path)
            time.sleep(CHECK_INTERVAL)
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main_loop()
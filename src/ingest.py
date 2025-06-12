import os
from uuid import uuid4
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from config import CONFIG, initialize_models, initialize_vector_store, setup_signal_handlers
import sys

logger = logging.getLogger(__name__)

# Track processed files to avoid duplicates
PROCESSED_FILES = set()

def ingest_file(file_path, vector_store):
    """Ingest a single PDF file into the vector store."""
    if not file_path.lower().endswith('.pdf'):
        logger.info(f"Skipping non-PDF file: {file_path}")
        return
    if file_path in PROCESSED_FILES:
        logger.info(f"Skipping already processed file: {file_path}")
        return

    try:
        logger.info(f"Starting to ingest file: {file_path}")
        loader = PyMuPDFLoader(file_path)
        loaded_documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["CHUNK_SIZE"],
            chunk_overlap=CONFIG["CHUNK_OVERLAP"],
            separators=["\n", " ", ""]
        )

        documents = text_splitter.split_documents(loaded_documents)
        uuids = [str(uuid4()) for _ in range(len(documents))]

        logger.info(f"Adding {len(documents)} documents to the vector store")
        vector_store.add_documents(documents=documents, ids=uuids)
        PROCESSED_FILES.add(file_path)
        logger.info(f"Finished ingesting file: {file_path}")

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")

def main():
    """Process PDF files in the data folder once."""
    setup_signal_handlers()
    models = initialize_models()
    vector_store = initialize_vector_store(models.embeddings_ollama)

    try:
        for filename in os.listdir(CONFIG["DATA_FOLDER"]):
            file_path = os.path.join(CONFIG["DATA_FOLDER"], filename)
            if not filename.startswith("_"):
                ingest_file(file_path, vector_store)
        logger.info("Completed processing all files in the data folder")
    except Exception as e:
        logger.error(f"Error in processing files: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
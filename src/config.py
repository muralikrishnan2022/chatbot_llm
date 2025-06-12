import os
import signal
import sys
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from models import Models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Environment variable defaults
CONFIG = {
    "DATA_FOLDER": os.getenv("DATA_FOLDER", "./pdfs"),
    "PERSIST_DIRECTORY": os.getenv("PERSIST_DIRECTORY", "./data/chroma_langchain_db"),
    "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", 1000)),
    "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", 50)),
    "CHECK_INTERVAL": int(os.getenv("CHECK_INTERVAL", 10)),
    "RETRIEVER_K": int(os.getenv("RETRIEVER_K", 3)),
    "COLLECTION_NAME": "documents"
}

def initialize_vector_store(embeddings):
    """Initialize the Chroma vector store."""
    try:
        vector_store = Chroma(
            collection_name=CONFIG["COLLECTION_NAME"],
            embedding_function=embeddings,
            persist_directory=CONFIG["PERSIST_DIRECTORY"]
        )
        logger.info("Successfully initialized Chroma vector store")
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        sys.exit(1)

def setup_signal_handlers():
    """Set up signal handlers for graceful termination."""
    def signal_handler(sig, frame):
        logger.info("Received termination signal. Exiting gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.info("Signal handlers registered")

def initialize_models():
    """Initialize the models."""
    try:
        models = Models()
        logger.info("Successfully initialized models")
        return models
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        sys.exit(1)
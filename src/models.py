import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
import logging

logger = logging.getLogger(__name__)

class Models:
    def __init__(self):
        try:
            # Load model names from environment variables with defaults
            embedding_model = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
            chat_model = os.getenv("CHAT_MODEL", "llama3:8b")

            # Initialize embeddings
            self.embeddings_ollama = OllamaEmbeddings(model=embedding_model)
            
            # Initialize chat model
            self.model_ollama = ChatOllama(
                model=chat_model,
                temperature=float(os.getenv("MODEL_TEMPERATURE", 0.3))
            )
            logger.info(f"Initialized models: embedding={embedding_model}, chat={chat_model}")
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise RuntimeError(f"Failed to initialize models: {str(e)}")
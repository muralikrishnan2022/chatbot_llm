import os
from langchain_ollama import OllamaEmbeddings, ChatOllama

class Models:
    def __init__(self):
        try:
            # Load model names from environment variables with defaults
            embedding_model = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
            chat_model = os.getenv("CHAT_MODEL", "llama3:8b")

            # Initialize embeddings
            self.embeddings_ollama = OllamaEmbeddings(model=embedding_model)
            
            # Initialize chat model with a more reasonable temperature
            self.model_ollama = ChatOllama(
                model=chat_model,
                temperature=float(os.getenv("MODEL_TEMPERATURE", 0.7))
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize models: {str(e)}")
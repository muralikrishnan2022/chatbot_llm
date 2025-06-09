import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from models import Models
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Allow CORS (important for frontend requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to your frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for chat input
class ChatRequest(BaseModel):
    messages: list  # List of dicts: {role: 'user' or 'assistant', content: str}

# Initialize models
try:
    models = Models()
    embeddings = models.embeddings_ollama
    llm = models.model_ollama
except Exception as e:
    print(f"Error initializing models: {str(e)}")
    sys.exit(1)

# Initialize vector store
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

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the question based only on the provided context."),
    ("human", "{context}\n\nQuestion: {input}")
])

# Retrieval chain
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 1))
try:
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
except Exception as e:
    print(f"Error setting up retrieval chain: {str(e)}")
    sys.exit(1)

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    try:
        # Extract the latest user message
        latest_user_msg = next((msg["content"] for msg in reversed(request.messages) if msg["role"] == "user"), None)
        if not latest_user_msg:
            raise HTTPException(status_code=400, detail="No user message found in request.")

        # Invoke retrieval chain
        result = retrieval_chain.invoke({"input": latest_user_msg})
        answer = result.get("answer", "No response found.")
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": answer
                }
            }]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

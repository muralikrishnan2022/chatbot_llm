import sys
import os
import shutil
import sqlite3
import bcrypt
import uvicorn
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from config import CONFIG, initialize_models, initialize_vector_stores, setup_signal_handlers
from ingest import ingest_file
from jose import JWTError, jwt
from datetime import datetime, timedelta
import unicodedata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure terminal encoding is UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Database setup
DB_PATH = os.path.join(CONFIG["DATA_FOLDER"], "users.db")

# JWT settings from .env
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Pydantic models
class UserRegister(BaseModel):
    username: str
    password: str

class User(BaseModel):
    username: str

class Query(BaseModel):
    question: str

class ToggleLanguage(BaseModel):
    use_hindi: bool


# Database initialization
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
        raise

# Dependency to get current user from token
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username

# Create JWT token
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Initialize models
models = initialize_models()

# Initialize database
initialize_database()

# Store user sessions and vector stores
user_sessions = {}



@app.post("/register")
async def register(user: UserRegister):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT username FROM users WHERE username = ?", (user.username,))
            if cursor.fetchone():
                logger.warning(f"Registration attempt for existing username: {user.username}")
                raise HTTPException(status_code=400, detail="Username already exists.")

            if '/' in user.username or '\\' in user.username or '..' in user.username:
                raise HTTPException(status_code=400, detail="Invalid username. Avoid special characters like /, \\, or ..")

            password_hash = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (user.username, password_hash)
            )
            conn.commit()
            logger.info(f"User {user.username} registered successfully")
            return {"message": "Registration successful."}
    except Exception as e:
        logger.error(f"Error registering user {user.username}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error registering user: {str(e)}")
    

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT password_hash FROM users WHERE username = ?", (form_data.username,))
            result = cursor.fetchone()
            if not result or not bcrypt.checkpw(form_data.password.encode('utf-8'), result[0]):
                logger.warning(f"Invalid login attempt for username: {form_data.username}")
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            session_id = str(uuid4())
            access_token = create_access_token(data={"sub": form_data.username})
            user_id = form_data.username.replace('/', '').replace('\\', '').replace('..', '_')
            
            # Initialize vector stores
            default_vector_store, user_vector_store, session_vector_store = initialize_vector_stores(
                models.embeddings_ollama, user_id, session_id
            )
            
            # Create upload directory
            upload_dir = os.path.join(CONFIG["DATA_FOLDER"], f"user_{user_id}")
            os.makedirs(upload_dir, exist_ok=True)
            
            # Store session data
            user_sessions[form_data.username] = {
                "session_id": session_id,
                "default_vector_store": default_vector_store,
                "user_vector_store": user_vector_store,
                "session_vector_store": session_vector_store,
                "upload_dir": upload_dir,
                "use_hindi": False,
                "documents_uploaded": False
            }
            
            # Initialize retrieval chains
            book_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer the question based only on the provided context from the book."
                           + (" Please provide in Hindi." if user_sessions[form_data.username]["use_hindi"] else "")),
                ("human", "{context}\n\nQuestion: {input}")
            ])
            user_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer the question based only on the provided context from user-uploaded documents."
                           + (" Please provide in Hindi." if user_sessions[form_data.username]["use_hindi"] else "")),
                ("human", "{context}\n\nQuestion: {input}")
            ])
            
            book_retriever = default_vector_store.as_retriever(search_kwargs={"k": CONFIG["RETRIEVER_K"]})
            book_combine_docs_chain = create_stuff_documents_chain(models.model_ollama, book_prompt)
            user_sessions[form_data.username]["book_retrieval_chain"] = create_retrieval_chain(book_retriever, book_combine_docs_chain)
            
            if session_vector_store:
                session_retriever = session_vector_store.as_retriever(search_kwargs={"k": CONFIG["RETRIEVER_K"]})
                session_combine_docs_chain = create_stuff_documents_chain(models.model_ollama, user_prompt)
                user_sessions[form_data.username]["session_retrieval_chain"] = create_retrieval_chain(session_retriever, session_combine_docs_chain)
            
            logger.info(f"User {form_data.username} authenticated with session_id: {session_id}")
            return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Error during authentication: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during authentication: {str(e)}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    session_data = user_sessions.get(current_user)
    if not session_data:
        raise HTTPException(status_code=400, detail="Session not found")
    
    upload_dir = session_data["upload_dir"]
    session_vector_store = session_data["session_vector_store"]
    
    try:
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Copied file to {file_path}")
        
        ingest_file(file_path, session_vector_store)
        session_data["documents_uploaded"] = True
        return {"message": "File uploaded successfully." if not session_data["use_hindi"] else "फ़ाइल सफलतापूर्वक अपलोड की गई।"}
    except Exception as e:
        logger.error(f"Error uploading file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/query/book")
async def query_book(query: Query, current_user: str = Depends(get_current_user)):
    session_data = user_sessions.get(current_user)
    if not session_data:
        raise HTTPException(status_code=400, detail="Session not found")
    
    if not query.question:
        logger.info("Empty query received")
        raise HTTPException(status_code=400, detail="Please enter a valid question." if not session_data["use_hindi"] else "कृपया एक मान्य प्रश्न दर्ज करें।")
    
    try:
        result = session_data["book_retrieval_chain"].invoke({"input": query.question})
        if not result.get("context"):
            logger.info("No relevant information found in the book")
            return {"answer": "No relevant information found in the book." if not session_data["use_hindi"] else "पुस्तक में कोई प्रासंगिक जानकारी नहीं मिली।"}
        
        logger.info(f"Retrieved {len(result['context'])} documents from book collection")
        answer = result["answer"]
        if session_data["user_vector_store"]:
            try:
                chat_text = f"User: {query.question}\nAssistant: {answer}"
                document = Document(page_content=chat_text, metadata={"type": "chat_history"})
                session_data["user_vector_store"].add_documents(documents=[document], ids=[str(uuid4())])
                logger.info("Stored chat history")
            except Exception as e:
                logger.error(f"Error storing chat history: {str(e)}")
        
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing book query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/query/documents")
async def query_documents(query: Query, current_user: str = Depends(get_current_user)):
    session_data = user_sessions.get(current_user)
    if not session_data or not session_data["session_vector_store"] or not session_data["documents_uploaded"]:
        raise HTTPException(status_code=400, detail="No documents available. Please upload a file first." if not session_data["use_hindi"] else "कोई दस्तावेज़ उपलब्ध नहीं हैं। कृपया पहले एक फ़ाइल अपलोड करें।")
    
    if not query.question:
        logger.info("Empty query received")
        raise HTTPException(status_code=400, detail="Please enter a valid question." if not session_data["use_hindi"] else "कृपया एक मान्य प्रश्न दर्ज करें।")
    
    try:
        result = session_data["session_retrieval_chain"].invoke({"input": query.question})
        if not result.get("context"):
            logger.info("No relevant information found in uploaded documents")
            return {"answer": "No relevant information found in uploaded documents." if not session_data["use_hindi"] else "अपलोड किए गए दस्तावेज़ों में कोई प्रासंगिक जानकारी नहीं मिली।"}
        
        logger.info(f"Retrieved {len(result['context'])} documents from session collection")
        answer = result["answer"]
        if session_data["user_vector_store"]:
            try:
                chat_text = f"User: {query.question}\nAssistant: {answer}"
                document = Document(page_content=chat_text, metadata={"type": "chat_history"})
                session_data["user_vector_store"].add_documents(documents=[document], ids=[str(uuid4())])
                logger.info("Stored chat history")
            except Exception as e:
                logger.error(f"Error storing chat history: {str(e)}")
        
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing user document query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/history")
async def view_history(current_user: str = Depends(get_current_user)):
    session_data = user_sessions.get(current_user)
    if not session_data or not session_data["user_vector_store"]:
        raise HTTPException(status_code=400, detail="No chat history available." if not session_data["use_hindi"] else "कोई चैट इतिहास उपलब्ध नहीं है।")
    
    try:
        results = session_data["user_vector_store"].get(where={"type": "chat_history"})
        if not results.get("documents"):
            return {"history": ["No chat history found." if not session_data["use_hindi"] else "कोई चैट इतिहास नहीं मिला।"]}
        
        logger.info("Displayed chat history")
        return {"history": [doc.strip() for doc in results["documents"]]}
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

@app.post("/toggle-language")
async def toggle_language(toggle: ToggleLanguage, current_user: str = Depends(get_current_user)):
    session_data = user_sessions.get(current_user)
    if not session_data:
        raise HTTPException(status_code=400, detail="Session not found")
    
    session_data["use_hindi"] = toggle.use_hindi
    try:
        book_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer the question based only on the provided context from the book."
                       + (" Please provide in Hindi." if toggle.use_hindi else "")),
            ("human", "{context}\n\nQuestion: {input}")
        ])
        user_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer the question based only on the provided context from user-uploaded documents."
                       + (" Please provide in Hindi." if toggle.use_hindi else "")),
            ("human", "{context}\n\n wheezingQuestion: {input}")
        ])
        
        book_retriever = session_data["default_vector_store"].as_retriever(search_kwargs={"k": CONFIG["RETRIEVER_K"]})
        book_combine_docs_chain = create_stuff_documents_chain(models.model_ollama, book_prompt)
        session_data["book_retrieval_chain"] = create_retrieval_chain(book_retriever, book_combine_docs_chain)
        
        if session_data["session_vector_store"]:
            session_retriever = session_data["session_vector_store"].as_retriever(search_kwargs={"k": CONFIG["RETRIEVER_K"]})
            session_combine_docs_chain = create_stuff_documents_chain(models.model_ollama, user_prompt)
            session_data["session_retrieval_chain"] = create_retrieval_chain(session_retriever, session_combine_docs_chain)
        
        logger.info(f"Switched to {'Hindi' if toggle.use_hindi else 'English'} mode")
        return {"message": f"Switched to {'Hindi' if toggle.use_hindi else 'English'} mode." if not toggle.use_hindi else f"{'हिंदी' if toggle.use_hindi else 'अंग्रेजी'} मोड में स्विच किया गया।"}
    except Exception as e:
        logger.error(f"Error switching language: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error switching language: {str(e)}")

@app.post("/logout")
async def logout(current_user: str = Depends(get_current_user)):
    session_data = user_sessions.get(current_user)
    if not session_data:
        raise HTTPException(status_code=400, detail="Session not found")
    
    try:
        if session_data["session_vector_store"]:
            session_data["session_vector_store"].delete_collection()
            logger.info(f"Deleted session vector store for session: {session_data['session_id']}")
        shutil.rmtree(session_data["upload_dir"], ignore_errors=True)
        logger.info(f"Deleted user upload directory: {session_data['upload_dir']}")
        user_sessions.pop(current_user, None)
        return {"message": "Logged out successfully." if not session_data["use_hindi"] else "सफलतापूर्वक लॉग आउट किया गया।"}
    except Exception as e:
        logger.error(f"Error during logout: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during logout: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main5:app", host="0.0.0.0", port=8000, reload=True)

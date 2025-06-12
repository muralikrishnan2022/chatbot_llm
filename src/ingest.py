import os
from uuid import uuid4
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging
from config import CONFIG, initialize_models, initialize_vector_stores, setup_signal_handlers
import sys
from PIL import Image, ImageEnhance
import pytesseract
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

# Track processed files to avoid duplicates
PROCESSED_FILES = set()

def preprocess_image(image):
    """Preprocess image to improve OCR accuracy."""
    try:
        # Convert to grayscale
        image = image.convert('L')
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        # Resize to ensure sufficient resolution (at least 300 DPI)
        width, height = image.size
        if width < 600 or height < 600:
            image = image.resize((int(width * 2), int(height * 2)), Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def extract_text_from_image(image_path):
    """Extract text from an image using Tesseract OCR with multiple attempts."""
    try:
        image = Image.open(image_path)
        image = preprocess_image(image)
        if image is None:
            logger.error(f"Preprocessing failed for {image_path}")
            return None

        # First attempt: Default OCR
        text = pytesseract.image_to_string(image, lang='eng')
        if text.strip():
            logger.info(f"Extracted text from {image_path}: {text[:100]}...")  # Log first 100 chars
            return text

        # Second attempt: Try with different page segmentation mode (PSM 6 for single block)
        logger.info(f"Retrying OCR on {image_path} with PSM 6")
        text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
        if text.strip():
            logger.info(f"Extracted text with PSM 6 from {image_path}: {text[:100]}...")
            return text

        logger.info(f"No text extracted from {image_path} after multiple attempts")
        return None
    except Exception as e:
        logger.error(f"Error extracting text from image {image_path}: {str(e)}")
        return None

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using PyMuPDFLoader or OCR if necessary."""
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        text = " ".join(doc.page_content for doc in documents if doc.page_content.strip())
        if text.strip():
            return documents
        # If no text is extracted, use OCR
        logger.info(f"No text extracted from {pdf_path} with PyMuPDFLoader, attempting OCR")
        images = convert_from_path(pdf_path)
        text = ""
        for i, image in enumerate(images):
            image = preprocess_image(image)
            if image is None:
                continue
            page_text = pytesseract.image_to_string(image, lang='eng')
            if page_text.strip():
                text += f"\nPage {i+1}:\n{page_text}"
        if text.strip():
            return [Document(page_content=text, metadata={"source": pdf_path})]
        logger.info(f"No text extracted from {pdf_path} via OCR")
        return []
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return []

def ingest_file(file_path, vector_store):
    """Ingest a PDF or image file into the vector store."""
    file_path = os.path.normpath(file_path)  # Normalize path for Windows
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_path in PROCESSED_FILES:
        logger.info(f"Skipping already processed file: {file_path}")
        return

    try:
        logger.info(f"Starting to ingest file: {file_path}")
        documents = []
        if file_extension == '.pdf':
            documents = extract_text_from_pdf(file_path)
        elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff']:
            text = extract_text_from_image(file_path)
            if text:
                documents = [Document(page_content=text, metadata={"source": file_path})]
            else:
                logger.info(f"No text extracted from image: {file_path}")
                return
        else:
            logger.info(f"Skipping unsupported file: {file_path}")
            return

        if not documents:
            logger.info(f"No documents to ingest from {file_path}")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["CHUNK_SIZE"],
            chunk_overlap=CONFIG["CHUNK_OVERLAP"],
            separators=["\n", " ", ""]
        )

        split_documents = text_splitter.split_documents(documents)
        uuids = [str(uuid4()) for _ in range(len(split_documents))]

        logger.info(f"Adding {len(split_documents)} documents to the vector store")
        vector_store.add_documents(documents=split_documents, ids=uuids)
        PROCESSED_FILES.add(file_path)
        logger.info(f"Finished ingesting file: {file_path}")

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")

def main():
    """Process files in the data folder once into the default vector store."""
    setup_signal_handlers()
    models = initialize_models()
    default_vector_store, _ = initialize_vector_stores(models.embeddings_ollama)

    try:
        for filename in os.listdir(CONFIG["DATA_FOLDER"]):
            file_path = os.path.join(CONFIG["DATA_FOLDER"], filename)
            if not filename.startswith("_"):
                ingest_file(file_path, default_vector_store)
        logger.info("Completed processing all files in the data folder")
    except Exception as e:
        logger.error(f"Error in processing files: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
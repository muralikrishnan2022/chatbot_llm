import sys
import unicodedata
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from transformers import MarianMTModel, MarianTokenizer
import logging
from config import CONFIG, initialize_models, initialize_vector_store, setup_signal_handlers

logger = logging.getLogger(__name__)

# Ensure terminal encoding is UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def main():
    """Main loop for user interaction."""
    setup_signal_handlers()
    models = initialize_models()
    vector_store = initialize_vector_store(models.embeddings_ollama)

    # Initialize translation model
    try:
        save_directory = "en-hi-model"
        tokenizer = MarianTokenizer.from_pretrained(save_directory)
        model = MarianMTModel.from_pretrained(save_directory, use_safetensors=True)
        logger.info("Initialized translation model")
    except Exception as e:
        logger.error(f"Error initializing translation model: {str(e)}")
        sys.exit(1)

    # Define the chat prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question based only on the provided context."),
        ("human", "{context}\n\nQuestion: {input}")
    ])

    # Set up the retrieval chain
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": CONFIG["RETRIEVER_K"]})
        combine_docs_chain = create_stuff_documents_chain(models.model_ollama, prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        logger.info("Initialized retrieval chain")
    except Exception as e:
        logger.error(f"Error setting up retrieval chain: {str(e)}")
        sys.exit(1)

    while True:
        query = input("User (or type 'q', 'quit', or 'exit' to end): ").strip()
        if query.lower() in ['q', 'quit', 'exit']:
            break
        if not query:
            logger.info("Empty query received")
            print("Assistant: Please enter a valid question.\n")
            print("सहायक: कृपया एक मान्य प्रश्न दर्ज करें।\n")
            continue

        try:
            result = retrieval_chain.invoke({"input": query})
            if not result.get("context"):
                logger.info("No relevant information found for query")
                print("Assistant: No relevant information found in the documents.\n")
                print("सहायक: दस्तावेज़ों में कोई प्रासंगिक जानकारी नहीं मिली।\n")
            else:
                answer = result["answer"]
                print("Assistant:", answer, "\n")
                inputs = tokenizer(str(answer), return_tensors="pt", padding=True)
                translated = model.generate(**inputs)
                decoded_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
                for text in decoded_texts:
                    normalized_text = unicodedata.normalize('NFC', text)
                    print(f"सहायक: {normalized_text}")
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print(f"Assistant: Error processing query: {str(e)}\n")

if __name__ == "__main__":
    main()
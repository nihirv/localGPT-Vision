import os
import uuid
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from models.indexer import index_documents
from models.retriever import retrieve_documents
from models.responder import generate_response
from byaldi import RAGMultiModalModel
from logger import get_logger
import threading
import gc

# Initialize FastAPI app
app = FastAPI(title="LocalGPT Vision API")
logger = get_logger(__name__)

# Configure folders
UPLOAD_FOLDER = 'uploaded_documents'
INDEX_FOLDER = os.path.join(os.getcwd(), '.byaldi')
STATIC_FOLDER = 'static'


# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Global variables
MODEL_NAME = 'vidore/colqwen2-v1.0'
model_lock = threading.Lock()  # Lock for thread-safe model operations
current_model = None

def get_or_create_model(file_index: str = None) -> RAGMultiModalModel:
    """Get existing model or create a new one if needed"""
    global current_model
    
    with model_lock:
        try:
            index_path = os.path.join(INDEX_FOLDER, file_index) if file_index else None
            
            # If we have a file_index and its index exists, load from index
            if file_index and os.path.exists(index_path):
                if current_model:
                    del current_model  # Release current model
                    gc.collect()  # Force garbage collection
                
                current_model = RAGMultiModalModel.from_index(
                    index_path=index_path,
                    device="cuda"
                )
                logger.info(f"Loaded model from existing index: {file_index}")
            
            # If no index exists or no file_index provided, create new model
            elif current_model is None:
                current_model = RAGMultiModalModel.from_pretrained(MODEL_NAME)
                logger.info("Created new base model")
                
                # If we have a file_index, index the documents
                if file_index and os.path.exists(os.path.join(UPLOAD_FOLDER, file_index)):
                    current_model.index(
                        input_path=os.path.join(UPLOAD_FOLDER, file_index),
                        index_name=file_index,
                        store_collection_with_index=True,
                        overwrite=True,
                        max_image_width=2048,  # Increased resolution
                        max_image_height=2048  # Increased resolution
                    )
                    logger.info(f"Indexed documents for {file_index}")
            
            return current_model
            
        except Exception as e:
            logger.error(f"Error in get_or_create_model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

def get_rag_model(file_index: str) -> RAGMultiModalModel:
    """Get RAG model for the given file index"""
    try:
        return get_or_create_model(file_index)
    except Exception as e:
        logger.error(f"Error in get_rag_model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.post("/index")
async def index_file(file: UploadFile = File(...)):
    """Index a file and return its file_index"""
    try:
        # Use filename as the index identifier
        file_index = file.filename
        session_folder = os.path.join(UPLOAD_FOLDER, file_index)
        index_path = os.path.join(INDEX_FOLDER, file_index)

        # Check if index already exists
        if os.path.exists(index_path):
            logger.info(f"Index already exists for {file_index}, loading existing model")
            model = get_or_create_model(file_index)
            return {
                "success": True,
                "file_index": file_index,
                "message": "Using existing index"
            }

        # If no index exists, create the upload folder and save the file
        os.makedirs(session_folder, exist_ok=True)
        file_path = os.path.join(session_folder, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Index using the model
        model = get_or_create_model(file_index)
        
        return {
            "success": True,
            "file_index": file_index,
            "message": "File indexed successfully"
        }

    except Exception as e:
        logger.error(f"Error indexing file: {str(e)}")
        # Clean up session folder if indexing fails
        if os.path.exists(session_folder):
            import shutil
            shutil.rmtree(session_folder)
        raise HTTPException(status_code=500, detail=f"Error indexing file: {str(e)}")

@app.post("/search")
async def search_documents(file_index: str, query: str, k: int = 20):
    """
    Search documents using the provided query
    """
    try:
        RAG = get_rag_model(file_index)
        # Ensure k is an integer
        k = int(k)
        retrieved_images = retrieve_documents(RAG, query, file_index, k)
        
        return {
            "success": True,
            "retrieved_images": retrieved_images
        }

    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@app.post("/generate")
@app.post("/generate")
async def generate(file_index: str, query: str, retrieved_images: List[str]):
    """
    Generate response using the provided retrieved images
    """
    try:
        RAG = get_rag_model(file_index)
        response_text, used_images = generate_response(
            retrieved_images,
            query,
            file_index,
            resized_height=512,
            resized_width=512,
            model_choice='gpt4'
        )
        return {
            "success": True,
            "response": response_text,
            "used_images": used_images
        }
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
        
@app.post("/search_and_generate")
async def search_and_generate(file_index: str, query: str, k: int = 20):
    try:
        # First search
        retrieved_docs = await search_documents(file_index, query, k)
        if not retrieved_docs["success"]:
            raise HTTPException(status_code=500, detail="Error retrieving documents")
            
        # Then generate using the retrieved images
        response = await generate(
            file_index=file_index,
            query=query,
            retrieved_images=retrieved_docs["retrieved_images"]
        )
        if not response["success"]:
            raise HTTPException(status_code=500, detail="Error generating response")
            
        return {
            "success": True,
            "retrieved_images": retrieved_docs["retrieved_images"],
            "response": response["response"],
            "used_images": response["used_images"]
        }
    except Exception as e:
        logger.error(f"Error in search and generate: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in search and generate: {str(e)}")

        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)

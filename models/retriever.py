import base64
import os
from PIL import Image
from io import BytesIO
from logger import get_logger
import hashlib
from typing import List

logger = get_logger(__name__)

def retrieve_documents(RAG, query: str, session_id: str, k: int = 20) -> List[str]:
    """
    Retrieves relevant documents based on the user query using Byaldi.

    Args:
        RAG (RAGMultiModalModel): The RAG model with the indexed documents.
        query (str): The user's query.
        session_id (str): The session ID to store images in per-session folder.
        k (int): The number of documents to retrieve.

    Returns:
        List[str]: A list of image filenames corresponding to the retrieved documents.
    """
    try:
        logger.info(f"Retrieving documents for query: {query}")
        # Ensure k is an integer
        k = int(k)
        results = RAG.search(query, k=k)
        images = []
        
        # Log the results structure for debugging
        logger.debug(f"Search results type: {type(results)}")
        if results:
            logger.debug(f"First result type: {type(results[0])}")
            logger.debug(f"First result attributes: {dir(results[0])}")
        
        for i, result in enumerate(results):
            try:
                # Check if result has base64 attribute and it's not None/empty
                if hasattr(result, 'base64') and result.base64:
                    image_data = base64.b64decode(result.base64)
                    image = Image.open(BytesIO(image_data))
                    logger.info(f"Image dimensions: {image.size}")  # width x height
                    
                    # Generate filename based on content
                    image_hash = hashlib.md5(image_data).hexdigest()
                    
                    # Save the retrieved image
                    session_images_folder = os.path.join('static', 'images', session_id)
                    os.makedirs(session_images_folder, exist_ok=True)
                    
                    image_filename = f"retrieved_{image_hash}.jpg"
                    image_path = os.path.join(session_images_folder, image_filename)
                    
                    if not os.path.exists(image_path):
                        # Save with best possible quality
                        image.save(
                            image_path,
                            "JPEG",
                            quality=100,
                            optimize=True,
                            subsampling=0
                        )
                        logger.debug(f"Saved image: {image_path}")
                    
                    # Store the relative path from the static folder
                    relative_path = os.path.join('images', session_id, image_filename)
                    images.append(relative_path)
                else:
                    logger.warning(f"Result {i} has no base64 data: {result}")
            except Exception as e:
                logger.error(f"Error processing image {i}: {str(e)}", exc_info=True)
                continue
        
        logger.info(f"Successfully retrieved {len(images)} images")
        return images
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
        raise
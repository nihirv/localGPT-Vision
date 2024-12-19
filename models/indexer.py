# models/indexer.py

import os
from byaldi import RAGMultiModalModel
from models.converters import convert_docs_to_pdfs
from logger import get_logger

logger = get_logger(__name__)

def index_documents(folder_path, index_name='document_index', index_path=None, indexer_model='vidore/colpali'):
    """
    Indexes documents in the specified folder using Byaldi.

    Args:
        folder_path (str): The path to the folder containing documents to index.
        index_name (str): The name of the index to create or update.
        index_path (str): The path where the index should be saved.
        indexer_model (str): The name of the indexer model to use.

    Returns:
        RAGMultiModalModel: The RAG model with the indexed documents.
    """
    try:
        logger.info(f"Starting document indexing in folder: {folder_path}")
        
        # Convert non-PDF documents to PDFs
        convert_docs_to_pdfs(folder_path)
        logger.info("Conversion of non-PDF documents to PDFs completed.")

        # Initialize the model
        if index_path and os.path.exists(index_path):
            RAG = RAGMultiModalModel.from_index(
                index_path=index_path,
                device="cuda"
            )
            logger.info(f"Loaded existing index from {index_path}")
        else:
            RAG = RAGMultiModalModel.from_pretrained(indexer_model)
            if RAG is None:
                raise ValueError(f"Failed to initialize RAGMultiModalModel with model {indexer_model}")
            logger.info("Created new RAG model instance")

            # Index the documents directly using Byaldi
            RAG.index(
                input_path=folder_path,
                index_name=index_name,
                store_collection_with_index=True,
                max_image_width=2048,  # Set high resolution limits
                max_image_height=2048,
            )
            logger.info(f"Indexed documents in {folder_path}")

            # Save the index if a path is provided
            if index_path:
                os.makedirs(os.path.dirname(index_path), exist_ok=True)
                RAG.save_index(index_path)
                logger.info(f"Saved index to {index_path}")

        return RAG

    except Exception as e:
        logger.error(f"Error during document indexing: {str(e)}")
        raise
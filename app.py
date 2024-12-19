import os
import uuid
import json
import time
from fastapi import FastAPI, Request, Form, File, UploadFile, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import SessionMiddleware
from starlette.middleware.sessions import SessionMiddleware
from typing import List, Optional
from markupsafe import Markup
from models.indexer import index_documents
from models.retriever import retrieve_documents
from models.responder import generate_response
from werkzeug.utils import secure_filename
from logger import get_logger
from byaldi import RAGMultiModalModel
import markdown

# Set the TOKENIZERS_PARALLELISM environment variable to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the FastAPI application
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your_secret_key")

# Initialize templates
templates = Jinja2Templates(directory="templates")

logger = get_logger(__name__)

# Configure upload folders
UPLOAD_FOLDER = 'uploaded_documents'
STATIC_FOLDER = 'static'
SESSION_FOLDER = 'sessions'
INDEX_FOLDER = os.path.join(os.getcwd(), '.byaldi')

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(SESSION_FOLDER, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")

# Initialize global variables
RAG_models = {}  # Dictionary to store RAG models per session
initialization_done = False

def load_rag_model_for_session(session_id: str):
    """
    Loads the RAG model for the given session_id from the index on disk.
    """
    index_path = os.path.join(INDEX_FOLDER, session_id)

    if os.path.exists(index_path):
        try:
            RAG = RAGMultiModalModel.from_index(index_path)
            RAG_models[session_id] = RAG
            logger.info(f"RAG model for session {session_id} loaded from index.")
        except Exception as e:
            logger.error(f"Error loading RAG model for session {session_id}: {e}")
    else:
        logger.warning(f"No index found for session {session_id}.")

def load_existing_indexes():
    """
    Loads all existing indexes from the .byaldi folder when the application starts.
    """
    global RAG_models
    if os.path.exists(INDEX_FOLDER):
        for session_id in os.listdir(INDEX_FOLDER):
            if os.path.isdir(os.path.join(INDEX_FOLDER, session_id)):
                load_rag_model_for_session(session_id)
    else:
        logger.warning("No .byaldi folder found. No existing indexes to load.")

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global initialization_done
    if not initialization_done:
        load_existing_indexes()
        initialization_done = True
        logger.info("Application initialized and indexes loaded.")

def get_session_id(request: Request) -> str:
    """Get or create session ID"""
    session = request.session
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

@app.get("/", response_class=RedirectResponse)
async def home():
    return RedirectResponse(url="/chat")

@app.get("/chat", response_class=HTMLResponse)
async def chat_get(request: Request):
    session_id = get_session_id(request)
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")

    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            session_data = json.load(f)
            chat_history = session_data.get('chat_history', [])
            session_name = session_data.get('session_name', 'Untitled Session')
            indexed_files = session_data.get('indexed_files', [])
    else:
        chat_history = []
        session_name = 'Untitled Session'
        indexed_files = []

    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "chat_history": chat_history,
            "session_name": session_name,
            "indexed_files": indexed_files
        }
    )

@app.post("/chat/upload")
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
):
    session_id = get_session_id(request)
    session_folder = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(session_folder, exist_ok=True)
    
    uploaded_files = []
    for file in files:
        if file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(session_folder, filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            uploaded_files.append(filename)
            logger.info(f"File saved: {file_path}")

    if uploaded_files:
        try:
            index_name = session_id
            index_path = os.path.join(INDEX_FOLDER, index_name)
            indexer_model = request.session.get('indexer_model', 'vidore/colpali')
            RAG = index_documents(session_folder, index_name=index_name, index_path=index_path, indexer_model=indexer_model)
            
            if RAG is None:
                raise ValueError("Indexing failed: RAG model is None")
                
            RAG_models[session_id] = RAG
            request.session['index_name'] = index_name
            request.session['session_folder'] = session_folder
            
            # Update session file
            session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
            if os.path.exists(session_file):
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
            else:
                session_data = {
                    'session_name': 'Untitled Session',
                    'chat_history': [],
                    'indexed_files': []
                }
            
            session_data['indexed_files'].extend(uploaded_files)
            with open(session_file, 'w') as f:
                json.dump(session_data, f)
                
            return JSONResponse({
                "success": True,
                "message": "Files indexed successfully.",
                "indexed_files": session_data['indexed_files']
            })
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"Error indexing files: {str(e)}"}
            )
    
    return JSONResponse(
        status_code=400,
        content={"success": False, "message": "No files were uploaded."}
    )

@app.post("/chat/query")
async def handle_query(
    request: Request,
    query: str = Form(...),
):
    session_id = get_session_id(request)
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")

    try:
        generation_model = request.session.get('generation_model', 'qwen')
        resized_height = request.session.get('resized_height', 280)
        resized_width = request.session.get('resized_width', 280)
        
        rag_model = RAG_models.get(session_id)
        if rag_model is None:
            logger.error(f"RAG model not found for session {session_id}")
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "RAG model not found for this session."}
            )
        
        retrieved_images = retrieve_documents(rag_model, query, session_id)
        logger.info(f"Retrieved images: {retrieved_images}")
        
        full_image_paths = [os.path.join(STATIC_FOLDER, img) for img in retrieved_images]
        response_text, used_images = generate_response(
            full_image_paths,
            query,
            session_id,
            resized_height,
            resized_width,
            generation_model
        )
        
        parsed_response = Markup(markdown.markdown(response_text))
        relative_images = [os.path.relpath(img, STATIC_FOLDER) for img in used_images]

        # Load and update chat history
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                session_data = json.load(f)
        else:
            session_data = {
                'session_name': query[:50] if len(query) <= 50 else query[:47] + "...",
                'chat_history': [],
                'indexed_files': []
            }

        session_data['chat_history'].append({"role": "user", "content": query})
        session_data['chat_history'].append({
            "role": "assistant",
            "content": parsed_response,
            "images": relative_images
        })

        with open(session_file, 'w') as f:
            json.dump(session_data, f)

        return JSONResponse({
            "success": True,
            "response": str(parsed_response),
            "images": relative_images
        })

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error processing query: {str(e)}"}
        )

@app.post("/settings")
async def update_settings(
    request: Request,
    generation_model: str = Form(...),
    indexer_model: str = Form(...),
    resized_height: int = Form(...),
    resized_width: int = Form(...),
):
    request.session['generation_model'] = generation_model
    request.session['indexer_model'] = indexer_model
    request.session['resized_height'] = resized_height
    request.session['resized_width'] = resized_width
    return JSONResponse({"success": True, "message": "Settings updated successfully"})

@app.get("/settings")
async def get_settings(request: Request):
    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "generation_model": request.session.get('generation_model', 'qwen'),
            "indexer_model": request.session.get('indexer_model', 'vidore/colpali'),
            "resized_height": request.session.get('resized_height', 280),
            "resized_width": request.session.get('resized_width', 280)
        }
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)
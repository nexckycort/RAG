from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama

MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

llm = Llama(
    model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=1024,  
    n_threads=4, 
    n_batch=256, 
)

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str 

class TextChunksRequest(BaseModel):
    chunks: List[str]

class PromptRequest(BaseModel):
    prompt: str

@app.get("/status")
async def read_root():
    return {"message": "ok"}

@app.post("/embed")
async def embed_text(request: TextRequest):
    try:
        embedding = model.encode(request.text).tolist()
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar el embedding: {str(e)}")

@app.post("/embed-chunks")
async def embed_chunks(request: TextChunksRequest):
    try:
        embeddings = [model.encode(chunk).tolist() for chunk in request.chunks]
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar los embeddings: {str(e)}")
    
@app.post("/ask")
async def generate_response(request: PromptRequest):
    try:
        output = llm(
            request.prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9
        )
        print("Output recibido:", output)

        return {"response": output["choices"][0]["text"].strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
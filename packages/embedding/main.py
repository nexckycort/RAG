from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
from fastapi.middleware.cors import CORSMiddleware

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
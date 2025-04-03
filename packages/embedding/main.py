from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/status")
def read_root():
    return {"message": "ok"}

class TextRequest(BaseModel):
    text: str

@app.post("/embed")
def embed_text(request: TextRequest):
    embedding = model.encode(request.text).tolist()
    return {"embedding": embedding}

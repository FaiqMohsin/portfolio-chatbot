# main.py - Chatbot API for Portfolio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cohere
import chromadb
from chromadb.utils import embedding_functions

# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────
API_KEY = "zZPz0NpP3lgzjnUMvqUQ0nrzbWpJxwk9cjEjNZUU"

cohere_client = cohere.ClientV2(api_key=API_KEY)

cohere_ef = embedding_functions.CohereEmbeddingFunction(
    api_key=API_KEY,
    model_name="embed-v4.0"
)

chroma_client = chromadb.EphemeralClient()
collection = chroma_client.create_collection(
    "faiq_info",
    embedding_function=cohere_ef
)

# ─────────────────────────────────────────
# INDEX faiq_info.txt ON STARTUP
# reads the file, chunks it, stores in chromadb
# this runs once when the server starts
# ─────────────────────────────────────────
def load_knowledge_base():
    with open("faiq_info.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # chunk into 500 char pieces with 50 char overlap
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + 500].strip()
        if chunk:
            chunks.append(chunk)
        start += 450

    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"Loaded {len(chunks)} chunks from faiq_info.txt")

load_knowledge_base()

# ─────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────
app = FastAPI(title="Faiq Chatbot API")

# CORS allows your portfolio website to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

# ─────────────────────────────────────────
# CHAT ENDPOINT
# receives question, retrieves relevant chunks,
# generates answer using cohere
# ─────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # retrieve relevant chunks from chromadb
    results = collection.query(
        query_texts=[request.question],
        n_results=min(4, collection.count())
    )

    chunks = results["documents"][0]
    context = "\n\n".join(chunks)

    # generate answer using cohere
    prompt = f"""You are Faiq Mohsin's personal AI assistant on his portfolio website.
Answer questions about Faiq using ONLY the context below.
Be friendly, professional, and concise.
If the answer is not in the context say "I don't have that information. You can contact Faiq directly at faiqmohsin7@gmail.com"

Context:
{context}

Question: {request.question}
Answer:"""

    response = cohere_client.chat(
        model="command-a-plus-05-2026",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = ""
    for block in response.message.content:
        if hasattr(block, 'text'):
            answer = block.text
            break

    return ChatResponse(answer=answer)

# ─────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "running", "chunks": collection.count()}
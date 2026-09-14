# main.py - Chatbot API for Portfolio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cohere
import os

# ─────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────
API_KEY = os.environ.get("COHERE_API_KEY", "")
cohere_client = cohere.ClientV2(api_key=API_KEY)

# load knowledge base at startup
knowledge_base = ""

def load_knowledge_base():
    global knowledge_base
    with open("faiq_info.txt", "r", encoding="utf-8") as f:
        knowledge_base = f.read()
    print(f"Loaded knowledge base: {len(knowledge_base)} characters")

load_knowledge_base()

# ─────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────
app = FastAPI(title="Faiq Chatbot API")

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
# sends full knowledge base as context
# simpler than RAG but works perfectly for
# a small knowledge base like faiq_info.txt
# ─────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    prompt = f"""You are Faiq Mohsin's personal AI assistant on his portfolio website.
Answer questions about Faiq using ONLY the context below.
Be friendly, professional, and concise.
If the answer is not in the context say "I don't have that information. You can contact Faiq directly at faiqmohsin7@gmail.com"

Context:
{knowledge_base}

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

@app.get("/health")
def health():
    return {"status": "running", "knowledge_base_chars": len(knowledge_base)}

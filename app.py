import os
import uuid
from typing import List, Optional

import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---- OpenAI (new SDK) ----
from openai import OpenAI

# ---- Pinecone (v4 SDK). If you must use the old client, see comment further below.
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "mini-rag")
PC_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PC_REGION = os.getenv("PINECONE_REGION", "us-east-1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536 dims
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY and PINECONE_API_KEY in .env")

# ---- Clients ----
oai = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create the index if missing (serverless)
def ensure_index():
    try:
        names = [i.name for i in pc.list_indexes()]
        if INDEX_NAME not in names:
            pc.create_index(
                name=INDEX_NAME,
                dimension=1536,           # for text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(cloud=PC_CLOUD, region=PC_REGION),
            )
    except Exception as e:
        raise RuntimeError(f"Could not ensure Pinecone index: {e}")

ensure_index()
index = pc.Index(INDEX_NAME)

# -------------- Utilities --------------

# Simple token-aware chunker (approx) to avoid giant passages
_tokenizer = tiktoken.get_encoding("cl100k_base")
def chunk_text(text: str, max_tokens: int = 300, overlap: int = 50) -> List[str]:
    tokens = _tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = _tokenizer.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def build_rag_prompt(query: str, contexts: List[str]) -> List[dict]:
    context_block = "\n\n---\n\n".join(contexts)
    system = (
        "You are a precise assistant. Use ONLY the provided context to answer. "
        "If the answer is not in the context, say you don't know."
    )
    user = f"Context:\n{context_block}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

# -------------- FastAPI models --------------
class Doc(BaseModel):
    id: Optional[str] = None
    text: str
    source: Optional[str] = None  # e.g., filename / URL

class IngestRequest(BaseModel):
    documents: List[Doc]

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

# -------------- FastAPI app --------------
app = FastAPI(title="Mini RAG API", version="1.0.0")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ingest")
def ingest(payload: IngestRequest):
    """
    Index documents. We chunk -> embed -> upsert into Pinecone.
    """
    vectors = []
    for doc in payload.documents:
        doc_id = doc.id or str(uuid.uuid4())
        chunks = chunk_text(doc.text)
        embeddings = embed_texts(chunks)
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vid = f"{doc_id}::chunk::{i}"
            vectors.append({
                "id": vid,
                "values": emb,
                "metadata": {
                    "text": chunk,
                    "doc_id": doc_id,
                    "source": doc.source or "user",
                    "chunk_index": i,
                }
            })

    # batch upserts to stay under payload limits
    BATCH = 100
    for i in range(0, len(vectors), BATCH):
        index.upsert(vectors=vectors[i:i+BATCH])

    return {"indexed": len(vectors), "docs": len(payload.documents)}

@app.post("/query", response_model=QueryResponse)
def query_api(req: QueryRequest):
    """
    1) embed query
    2) retrieve from Pinecone
    3) compose prompt
    4) answer with sources
    """
    try:
        q_emb = embed_texts([req.query])[0]
        res = index.query(
            vector=q_emb,
            top_k=req.top_k,
            include_values=False,
            include_metadata=True,
        )
        matches = res.matches or []
        contexts = []
        srcs = []
        for m in matches:
            meta = m.metadata or {}
            contexts.append(meta.get("text", ""))
            srcs.append({
                "id": m.id,
                "doc_id": meta.get("doc_id"),
                "source": meta.get("source"),
                "score": float(m.score) if hasattr(m, "score") else None,
                "chunk_index": meta.get("chunk_index"),
            })

        if not contexts:
            return QueryResponse(answer="I don't know.", sources=[])

        messages = build_rag_prompt(req.query, contexts)
        chat = oai.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.0)
        answer = chat.choices[0].message.content.strip()

        return QueryResponse(answer=answer, sources=srcs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- Notes for old Pinecone client ----------------
# If you must use pinecone-client==2.x:
#   import pinecone
#   pinecone.init(api_key=PINECONE_API_KEY)
#   if INDEX_NAME not in pinecone.list_indexes():
#       pinecone.create_index(INDEX_NAME, dimension=1536, metric="cosine", pod_type="p1")
#   index = pinecone.Index(INDEX_NAME)

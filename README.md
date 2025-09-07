# Mini RAG with Pinecone and OpenAI

This project demonstrates how to build a lightweight Retrieval-Augmented Generation (RAG) system using **OpenAI embeddings**, **Pinecone** for vector storage, and a local API for ingesting and querying documents.

## Setup

1. Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
```

2. Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcn-...
PINECONE_INDEX=mini-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
```

3. Start the local server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```


## Usage

### Ingest Documents

Send text documents to the vector store for embedding and indexing:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "id": "doc-drupal",
        "source": "notes/drupal.txt",
        "text": "Drupal is a modular PHP CMS. Progressive decoupling lets you enhance specific sections with React or Vue while keeping editorial UX."
      },
      {
        "id": "doc-rag",
        "source": "notes/rag.txt",
        "text": "Retrieval-Augmented Generation (RAG) fetches relevant chunks from a vector database and feeds them to an LLM to ground answers."
      }
    ]
  }'
```


### Query the Index

Ask natural language questions and retrieve context-aware answers from your ingested documents:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is progressive decoupling in Drupal?",
    "top_k": 4
  }'
```


***

## Notes

- Use Thunder Client, Postman, or VS Code REST Client if you prefer a GUI over `curl`.
- Adjust `top_k` in the query to control how many relevant results are returned.
- You can expand the ingestion step with more documents to build up your knowledge base.



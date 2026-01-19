# API reference

This document describes the REST API for the Ethereum Protocol Intelligence System.

## Starting the server

```bash
# Development (with auto-reload)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`.

## Interactive documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### Health check

Check if the API is running.

```
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2026-01-15T12:00:00Z"
}
```

**Example:**

```bash
curl http://localhost:8000/health
```

---

### System statistics

Get statistics about the indexed corpus.

```
GET /stats
```

**Response:**

```json
{
  "documents": 881,
  "chunks": 8522,
  "last_indexed": "2026-01-15T11:30:00Z",
  "git_commit": "d154dfa2"
}
```

**Example:**

```bash
curl http://localhost:8000/stats
```

---

### Query

Submit a question and get a response.

```
POST /query
```

**Request body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | The question to answer |
| `mode` | string | No | `"simple"` | Query mode: `simple`, `cited`, `validated` |
| `top_k` | integer | No | `5` | Number of chunks to retrieve |

**Response:**

```json
{
  "query": "What is EIP-1559?",
  "response": "EIP-1559 is a significant Ethereum Improvement Proposal...",
  "mode": "simple",
  "sources": [
    {
      "document_id": "eip-1559",
      "section": "Abstract",
      "similarity": 0.72
    }
  ],
  "tokens": {
    "input": 1256,
    "output": 268
  },
  "validation": null
}
```

**Validation response (when mode=validated):**

```json
{
  "query": "What is the base fee?",
  "response": "The base fee is...",
  "mode": "validated",
  "sources": [...],
  "tokens": {...},
  "validation": {
    "total_claims": 5,
    "supported": 3,
    "weak": 1,
    "unsupported": 1,
    "contradicted": 0,
    "support_ratio": 0.6,
    "is_trustworthy": false,
    "flagged_claims": [
      {
        "text": "The base fee adjusts every block",
        "support_level": "WEAK",
        "confidence": 0.45
      }
    ]
  }
}
```

**Examples:**

```bash
# Simple query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is EIP-1559?"}'

# Cited query with more context
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do blob transactions work?", "mode": "cited", "top_k": 10}'

# Validated query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the base fee?", "mode": "validated"}'
```

---

### Get EIP

Get metadata and chunks for a specific EIP.

```
GET /eip/{number}
```

**Path parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `number` | integer | EIP number |

**Response:**

```json
{
  "document_id": "eip-1559",
  "eip_number": 1559,
  "title": "Fee market change for ETH 1.0 chain",
  "status": "Final",
  "type": "Standards Track",
  "category": "Core",
  "author": "Vitalik Buterin, Eric Conner, ...",
  "created": "2019-04-13",
  "requires": [2930],
  "chunks": [
    {
      "chunk_id": "abc123",
      "section_path": "Abstract",
      "content": "This EIP introduces a transaction pricing mechanism...",
      "token_count": 156
    }
  ]
}
```

**Example:**

```bash
curl http://localhost:8000/eip/1559
```

---

### Search

Search for relevant chunks without generating a response.

```
GET /search
```

**Query parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `q` | string | Yes | - | Search query |
| `top_k` | integer | No | `5` | Number of results |

**Response:**

```json
{
  "query": "base fee",
  "results": [
    {
      "document_id": "eip-1559",
      "section_path": "Specification",
      "content": "The base fee is burned...",
      "similarity": 0.78,
      "token_count": 234
    },
    {
      "document_id": "eip-1559",
      "section_path": "Abstract",
      "content": "This EIP introduces a transaction pricing...",
      "similarity": 0.65,
      "token_count": 156
    }
  ]
}
```

**Example:**

```bash
curl "http://localhost:8000/search?q=base%20fee&top_k=10"
```

---

## Error responses

All endpoints return standard error responses:

```json
{
  "detail": "Error message describing the issue"
}
```

**HTTP status codes:**

| Code | Meaning |
|------|---------|
| `200` | Success |
| `400` | Bad request (invalid parameters) |
| `404` | Not found (EIP doesn't exist) |
| `500` | Internal server error |
| `503` | Service unavailable (database connection failed) |

**Example error:**

```bash
curl http://localhost:8000/eip/99999
```

```json
{
  "detail": "EIP-99999 not found"
}
```

---

## Python client example

```python
import httpx

class EthProtocolClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.Client()

    def query(
        self,
        query: str,
        mode: str = "simple",
        top_k: int = 5
    ) -> dict:
        response = self.client.post(
            f"{self.base_url}/query",
            json={"query": query, "mode": mode, "top_k": top_k}
        )
        response.raise_for_status()
        return response.json()

    def get_eip(self, number: int) -> dict:
        response = self.client.get(f"{self.base_url}/eip/{number}")
        response.raise_for_status()
        return response.json()

    def search(self, query: str, top_k: int = 5) -> dict:
        response = self.client.get(
            f"{self.base_url}/search",
            params={"q": query, "top_k": top_k}
        )
        response.raise_for_status()
        return response.json()

# Usage
client = EthProtocolClient()

# Query
result = client.query("What is EIP-1559?", mode="cited")
print(result["response"])

# Get EIP metadata
eip = client.get_eip(1559)
print(f"{eip['title']} - {eip['status']}")

# Search
results = client.search("blob transactions")
for r in results["results"]:
    print(f"{r['document_id']}: {r['similarity']:.2f}")
```

---

## Rate limiting

The API does not implement rate limiting by default. For production deployments, consider:

- Using a reverse proxy (nginx, Caddy) with rate limiting
- Adding FastAPI middleware for rate limiting
- Implementing API keys for access control

---

## Authentication

The API is unauthenticated by default. For production, consider adding:

```python
# Example: API key authentication
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.environ.get("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

---

## CORS

Cross-Origin Resource Sharing is configured for development. Adjust for production:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],  # Specific origins in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

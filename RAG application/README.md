# RAG Financials (BSE/NSE) — Local Python + Qdrant + MCP + LangGraph

This project demonstrates a local-only Retrieval-Augmented Generation (RAG) workflow over quarterly results/financial statements (PDFs) for BSE/NSE-listed companies. It uses Qdrant (local container) as the vector DB, an MCP server to expose retrieval as a tool, and a LangGraph-based agent console app with streaming via Azure OpenAI.

## Tech Stack
- Python 3.11+
- Vector DB: Qdrant (local container)
- Orchestration: LangGraph (on top of LangChain components)
- Embeddings: Azure OpenAI embeddings
- Parsing/Chunking: LangChain loaders + RecursiveCharacterTextSplitter
- MCP: `mcp` Python SDK (stdio server)
- Console streaming: Ollama streaming + Rich

## Why LangGraph vs LangChain
- LangGraph: Best for agentic, stateful, tool-using workflows with clear control over nodes/edges and loops. More predictable and testable for interview demos.
- LangChain: Still used here for components (loaders, splitters, embeddings); LangGraph composes these into a robust agent.

## Project Structure
- apps/
  - agent_console/main.py — LangGraph agent that connects to MCP server and streams answers locally via Ollama
  - ingest/ingest.py — CLI to load PDFs, chunk, embed (sentence-transformers), and upsert to Qdrant
- servers/
  - mcp_financials/server.py — MCP stdio server exposing `semantic_search`
- src/
  - config/settings.py — env config (local-only)
  - rag/embeddings.py — sentence-transformers helpers
  - rag/chunking.py — chunking helpers
  - rag/vector_store.py — Qdrant helpers
- data/
  - raw/ — place quarterly PDF files here
  - processed/ — optional intermediate outputs
- scripts/ — (reserved for future utilities)

## Setup
1. Create and activate a virtual environment.

```bash
python -m venv .venv
. .venv/Scripts/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Create `.env` from example and fill values.

```bash
copy .env.example .env
# Set Azure and Qdrant vars:
# AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_EMBED_DEPLOYMENT
```

4. Start local services and add data.

```bash
# Qdrant (Podman Desktop)
podman run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest

# Azure OpenAI: configure Azure AD auth
# AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com
# AZURE_OPENAI_API_VERSION=2024-08-01-preview
# AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
# Authentication uses DefaultAzureCredential (Azure CLI login, VS Code, Managed Identity, etc.)
```

Add quarterly result PDFs to `data/raw`.

## Ingest to Pinecone
Embeds chunks via Azure OpenAI embeddings and upserts to Qdrant (creates the collection if missing).

```bash
python apps/ingest/ingest.py
```

### Chunking A/B configuration
You can configure chunking via environment variables and tag each ingestion run with a `run_id`:

```powershell
# Recursive chunking
$env:CHUNKING_MODE = "recursive"
$env:CHUNK_SIZE = "1200"
$env:CHUNK_OVERLAP = "200"
$env:RUN_ID = "rec_1200_200_v1"
python apps/ingest/ingest.py --chunking recursive

# Semantic chunking
$env:CHUNKING_MODE = "semantic"
$env:SEMANTIC_THRESHOLD = "0.4"
$env:RUN_ID = "sem_thr0_4_v1"
python apps/ingest/ingest.py --chunking semantic
```

Each upsert stores payload metadata on the chunk:
- `chunking.strategy`: `recursive` or `semantic`
- `chunking.chunk_size` / `chunking.chunk_overlap` or `chunking.threshold`
- `run_id`: identifier for the ingestion run
- `chunk_index`: index within the run

### Querying with filters
Filter by `run_id` or `chunking.strategy`:

```python
from src.rag.embeddings import embed_query
from src.rag.vector_store import query_embeddings

embedding = embed_query("What is the fiscal year revenue?")
res = query_embeddings(embedding, top_k=5, filter={"run_id": "rec_1200_200_v1"})
```

## Evaluation Harness
Compare retrieval results across two ingested runs defined in `.env` as `EVAL_RUN_A` and `EVAL_RUN_B`:

```powershell
$env:EVAL_RUN_A = "rec_1200_200_v1"
$env:EVAL_RUN_B = "sem_thr0_4_v1"
python apps/eval/compare_chunking.py --prompts apps/eval/prompts.txt --top-k 5 --out apps/eval/results.csv
```

Outputs a CSV per prompt with metrics: average score, overlap@k, and the retrieved IDs.

## Run Console Agent (Azure OpenAI streaming)
Launches an interactive console that:
- connects to the MCP server (stdio)
- calls `semantic_search` to retrieve context from Qdrant
- streams the final answer from your Azure OpenAI deployment

```bash
python apps/agent_console/main.py

# If using Azure CLI for auth, login before running:
# az login
```

## Notes
- You can later swap the console with a UI; the streaming API is already in place.

## Next Steps
- Add XLSX/CSV loaders and metadata extraction for symbol/period
- Add query rewriting (HyDE) and re-ranking
- Evaluate with RAGAS on a validation set
- Add unit tests and typed state for LangGraph

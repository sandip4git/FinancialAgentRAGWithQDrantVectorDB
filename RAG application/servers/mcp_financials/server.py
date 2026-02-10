import os
import sys

# Ensure project root on sys.path so 'src' imports work when launched from other cwd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)
import asyncio
from typing import Any, Dict
try:
    from fastmcp import FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP
from src.rag.embeddings import embed_query
from src.rag.vector_store import query_embeddings

app = FastMCP("financials-mcp")

@app.tool()
async def semantic_search(
    query: str,
    top_k: int = 5,
    symbol: str | None = None,
    period: str | None = None,
    runid: str | None = None,
) -> Dict[str, Any]:
    """
    Semantic search over financial documents in Pinecone.
    - query: natural language query
    - top_k: number of results
    - symbol: optional stock symbol filter (e.g., RELIANCE, TCS)
    - period: optional quarter/year filter (e.g., Q1FY25)
    Returns matches with metadata and scores.
    """
    emb = embed_query(query)
    flt: Dict[str, Any] = {}
    if symbol:
        flt["symbol"] = symbol
    if period:
        flt["period"] = period
    if runid:
        flt["runid"] = runid
    res = query_embeddings(emb, top_k=top_k, filter=flt or None)
    return {"matches": res["matches"]}

if __name__ == "__main__":
    # Run as stdio MCP server
    try:
        app.run()
    except* Exception as eg:
        import traceback
        print("MCP server failed to start or crashed.", file=sys.stderr)
        # Print inner exceptions for clearer diagnostics
        for ex in getattr(eg, "exceptions", [eg]):
            traceback.print_exception(ex)

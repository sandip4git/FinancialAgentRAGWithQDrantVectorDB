import asyncio
import json
import os
import sys
from typing import Any, Dict, List, TypedDict
import argparse
from rich.console import Console
from openai import AsyncAzureOpenAI
from azure.identity import DefaultAzureCredential
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from langgraph.graph import StateGraph
from azure.identity import DefaultAzureCredential
 

# Ensure project root on sys.path for src imports
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.config.settings import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
)
from src.rag.embeddings import embed_query
from src.rag.vector_store import query_embeddings, list_run_ids

console = Console()

class Citation(TypedDict, total=False):
    source: str
    page: int | None
    text: str
    score: float | None
    id: str

class AgentState(TypedDict):
    question: str
    contexts: List[str]
    citations: List[Citation]
    print_citations: bool
    answer: str

def get_azure_token():
    """Get Azure AD token for OpenAI"""
    credential = DefaultAzureCredential()
    token = credential.get_token("https://cognitiveservices.azure.com/.default")
    return token.token



async def retrieve(state: AgentState) -> AgentState:
    question = state["question"]
    # Start MCP stdio client to the server process
    server_cmd = [
        "python",
        os.path.join(ROOT, "servers", "mcp_financials", "server.py"),
    ]
    server_params = StdioServerParameters(command=server_cmd[0], args=server_cmd[1:], cwd=ROOT)
    async with stdio_client(server=server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize MCP session before listing tools/calling them
            await session.initialize()
            tools = await session.list_tools()
            tool = next((t for t in tools.tools if t.name == "semantic_search"), None)
            if tool is None:
                raise RuntimeError("MCP server missing 'semantic_search' tool")
            result = await session.call_tool("semantic_search", {"query": question, "top_k": 6})
            contexts: List[str] = []
            citations: List[Citation] = []
            for item in result.content:
                # FastMCP returns typed content; handle json or text
                if getattr(item, "type", None) == "json" and hasattr(item, "json"):
                    data = item.json
                else:
                    try:
                        data = json.loads(getattr(item, "text", "") or "{}")
                    except Exception:
                        data = {}
                matches = data.get("matches", []) if isinstance(data, dict) else []
                for m in matches:
                    meta = m.get("metadata", {}) if isinstance(m, dict) else {}
                    txt = meta.get("text")
                    if txt:
                        contexts.append(txt)
                        citations.append({
                            "source": meta.get("source", "unknown"),
                            "page": meta.get("page", None),
                            "text": txt,
                            "score": m.get("score", None),
                            "id": m.get("id", "")
                        })
    state["contexts"] = [c for c in contexts if c][:5]
    state["citations"] = citations[:5]
    return state

async def retrieve_for_run(state: AgentState, run_id: str, top_k: int = 6) -> AgentState:
    question = state["question"]
    emb = embed_query(question)
    res = query_embeddings(emb, top_k=top_k, filter={"run_id": run_id})
    contexts: List[str] = []
    citations: List[Citation] = []
    for m in res.get("matches", []):
        meta = m.get("metadata", {}) if isinstance(m, dict) else {}
        txt = meta.get("text")
        if txt:
            contexts.append(txt)
            citations.append({
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", None),
                "text": txt,
                "score": m.get("score", None),
                "id": m.get("id", "")
            })
    state["contexts"] = [c for c in contexts if c][:5]
    state["citations"] = citations[:5]
    return state

async def generate(state: AgentState) -> AgentState:
    # Acquire Azure AD token via default credentials (VS Code login, Azure CLI, Managed Identity, etc.)
    def get_azure_token() -> str:
        cred = DefaultAzureCredential()
        token = cred.get_token("https://cognitiveservices.azure.com/.default")
        return token.token

    client = AsyncAzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_ad_token_provider=get_azure_token,
    )
    question = state["question"]
    ctx = "\n\n".join(state.get("contexts", []))
    system = "You are a financial analyst. Answer using the provided context. If uncertain, say so."
    user = f"Question: {question}\n\nContext:\n{ctx}"

    console.print("\n[bold]Answer:[/bold]", end=" ")
    chunks = []
    stream = await client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,  # Azure: use deployment name
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=True        
    )
    async for event in stream:
        delta = (event.choices[0].delta.content or "") if event.choices and event.choices[0].delta else ""
        if delta:
            chunks.append(delta)
            console.print(delta, end="", soft_wrap=True)
    console.print()
    state["answer"] = "".join(chunks)
    # Print citations with source, page, and exact text snippet if enabled
    if state.get("print_citations", True):
        cits = state.get("citations", [])
        if cits:
            console.print("\n[bold]Citations:[/bold]")
            for i, c in enumerate(cits, start=1):
                src = c.get("source", "unknown")
                page = c.get("page")
                txt = c.get("text", "")
                snippet = txt if txt else ""
                if page is not None:
                    console.print(f"[{i}] {src} (page {page}) — {snippet}")
                else:
                    console.print(f"[{i}] {src} — {snippet}")
    return state

async def run_once(question: str, print_citations: bool = True):
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    app = graph.compile()
    out = await app.ainvoke({"question": question, "contexts": [], "citations": [], "print_citations": print_citations, "answer": ""})
    return out

async def run_for_all_runs(question: str, top_k: int = 6, print_citations: bool = True):
    runs = list_run_ids()
    if not runs:
        console.print("[yellow]No run_id values found in vector store.[/yellow]")
        return
    for rid in runs:
        console.print(f"\n[bold cyan]Run:[/bold cyan] {rid}")
        state: AgentState = {"question": question, "contexts": [], "citations": [], "print_citations": print_citations, "answer": ""}
        try:
            state = await retrieve_for_run(state, run_id=rid, top_k=top_k)
            await generate(state)
        except Exception as e:
            console.print(f"[red]Error for run {rid}:[/red] {e}")

async def run_for_runid(question: str, run_id: str, top_k: int = 6, print_citations: bool = True):
    console.print(f"\n[bold cyan]Run:[/bold cyan] {run_id}")
    state: AgentState = {"question": question, "contexts": [], "citations": [], "print_citations": print_citations, "answer": ""}
    state = await retrieve_for_run(state, run_id=run_id, top_k=top_k)
    await generate(state)

async def main():
    parser = argparse.ArgumentParser(description="Financials RAG Agent Console")
    parser.add_argument("--runid", type=str, default=None, help="Run only for the specified run id")
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", "6")), help="Number of results to retrieve per run")
    parser.add_argument("--print-citations", dest="print_citations", action="store_true", help="Print citations after the answer")
    parser.add_argument("--no-citations", dest="print_citations", action="store_false", help="Disable citation printing")
    parser.set_defaults(print_citations=True)
    args = parser.parse_args()

    console.print("[bold green]Financials RAG Agent[/bold green]")
    # use_file = os.getenv("USE_PROMPTS_FILE", "true").lower() in ("1", "true", "yes")
    # if use_file and os.path.isfile(prompts_file):
    #     with open(prompts_file, "r", encoding="utf-8") as f:
    #         prompts = [line.strip() for line in f if line.strip()]
    #     for q in prompts:
    #         console.print(f"\n[bold]Question:[/bold] {q}")
    #         await run_for_all_runs(q, top_k=top_k)
    #     return
    # console.print("Type your question (or 'exit'):")
    while True:
        q = input("\n> ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        try:
            if args.runid:
                await run_for_runid(q, run_id=args.runid, top_k=args.top_k, print_citations=args.print_citations)
            else:
                await run_for_all_runs(q, top_k=args.top_k, print_citations=args.print_citations)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")

if __name__ == "__main__":
    asyncio.run(main())

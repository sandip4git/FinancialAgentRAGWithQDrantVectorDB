import asyncio
import os
import sys
import json
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
import traceback

# Resolve project root
ROOT = os.path.dirname(os.path.dirname(__file__))

async def main():
    server_path = os.path.join(ROOT, "servers", "mcp_financials", "server.py")
    server = StdioServerParameters(command="python", args=[server_path], cwd=ROOT)
    # Capture server stderr to help diagnose startup/runtime issues
    err_path = os.path.join(ROOT, "scripts", "server_stderr.log")
    with open(err_path, "w", encoding="utf-8") as errlog:
        try:
            async with stdio_client(server=server, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    # Perform protocol initialization handshake before sending requests
                    await session.initialize()
                    tools = await session.list_tools()
                    print("Tools:", [t.name for t in tools.tools])
                    if any(t.name == "semantic_search" for t in tools.tools):
                        result = await session.call_tool("semantic_search", {"query": "What is net income tax paid ?", "top_k": 5})
                        payloads = []
                        for item in result.content:
                            data = None
                            if getattr(item, "type", None) == "json" and hasattr(item, "json"):
                                data = item.json
                            else:
                                try:
                                    data = json.loads(getattr(item, "text", "") or "{}")
                                except Exception:
                                    data = None
                            if isinstance(data, dict):
                                payloads.append(data)
                        print("semantic_search response:")
                        print(json.dumps(payloads, indent=2))
        except Exception:
            print("Encountered an error starting or communicating with MCP server.")
            traceback.print_exc()
            try:
                with open(err_path, "r", encoding="utf-8") as f:
                    tail = f.read()
                    print("--- server stderr ---")
                    print(tail[-2000:])
            except Exception:
                pass

if __name__ == "__main__":
    asyncio.run(main())

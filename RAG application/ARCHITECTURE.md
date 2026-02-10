@startuml
title Financials RAG Application Architecture

skinparam componentStyle rectangle
skinparam shadowing false

' Actors and core runtime
actor User
component "Agent Console\napps/agent_console/main.py" as Agent
component "FastMCP Server\nservers/mcp_financials/server.py" as MCP
database "Qdrant Vector DB" as Qdrant
component "Azure OpenAI\nChat Completions" as Chat
component "Azure OpenAI\nEmbeddings\nsrc/rag/embeddings.py" as Emb

' Ingestion pipeline
package "Ingestion" {
  collections "PDFs\ndata/raw" as PDFs
  component "Ingest\napps/ingest/ingest.py" as Ingest
  component "Chunking\nsrc/rag/chunking.py\n- RecursiveCharacterTextSplitter\n- SemanticChunker" as Chunk
  component "Azure OpenAI\nEmbeddings" as Emb2
}

' Retrieval grouping
package "Retrieval" {
  component "Query Points\nfilter by run_id, strategy" as QueryPoints
}

' Flows: user → agent → mcp → embeddings → qdrant
User --> Agent : Questions / Prompts
Agent --> MCP : MCP stdio
MCP --> Emb : embed_query()
Emb --> QueryPoints
QueryPoints --> Qdrant
Qdrant --> MCP : matches + payload\n(text, source, page, run_id)

' Agent assembles context and streams answer
Agent --> Chat : contexts + citations
Chat --> Agent : Answer + Citations

' Ingestion flows
PDFs --> Ingest
Ingest --> Chunk : Chunking Mode
Chunk --> Emb2 : texts
Emb2 --> Qdrant : Upsert Points\npayload(run_id,strategy,params,\nchunk_index,text,source,page)

' Notes and config
note right of Qdrant
- Single collection
- Payload indices: chunking.strategy, run_id
- Batched upserts (wait=true)
- Optional gRPC & timeouts
end note

note bottom of Agent
Reads prompts.txt, iterates all run_id,\nqueries per run for side-by-side answers
end note

@enduml
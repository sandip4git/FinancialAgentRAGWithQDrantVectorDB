import os
from dotenv import load_dotenv

load_dotenv()

# Vector DB selection
VECTOR_DB = os.getenv("VECTOR_DB", "qdrant").lower()

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "financials-index")

# Pinecone local (optional)
PINECONE_HOST = os.getenv("PINECONE_HOST", "localhost")
PINECONE_PORT = int(os.getenv("PINECONE_PORT", "8080"))
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "financials-index")

# Embeddings

# Azure OpenAI (LLM)
# Azure OpenAI Embeddings
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://<<your-end-point>>.openai.azure.com")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o4-mini-3")

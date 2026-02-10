from typing import List
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from src.config.settings import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBED_DEPLOYMENT,
)

_client: AzureOpenAI | None = None
_embed_dim: int | None = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        def get_azure_token() -> str:
            cred = DefaultAzureCredential()
            token = cred.get_token("https://cognitiveservices.azure.com/.default")
            return token.token

        _client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_ad_token_provider=get_azure_token,
        )
    return _client


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    client = _get_client()
    resp = client.embeddings.create(model=AZURE_OPENAI_EMBED_DEPLOYMENT, input=texts)
    vectors = [d.embedding for d in resp.data]
    global _embed_dim
    if vectors and _embed_dim is None:
        _embed_dim = len(vectors[0])
    return vectors


def embed_query(text: str) -> List[float]:
    client = _get_client()
    resp = client.embeddings.create(model=AZURE_OPENAI_EMBED_DEPLOYMENT, input=[text])
    vec = resp.data[0].embedding
    global _embed_dim
    if _embed_dim is None:
        _embed_dim = len(vec)
    return vec


def embedding_dimension() -> int:
    global _embed_dim
    if _embed_dim is not None:
        return _embed_dim
    # Fallback probe
    vec = embed_query("dimension probe")
    _embed_dim = len(vec)
    return _embed_dim

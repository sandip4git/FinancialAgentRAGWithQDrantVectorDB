from typing import List, Dict, Any, Iterable, cast
import os
from uuid import uuid4
import enum

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
    PayloadSchemaType,
)

from src.config.settings import VECTOR_DB, QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION

# Abstraction over local vector store (Qdrant now; Pinecone local optional later)

_def_client: QdrantClient | None = None


def get_qdrant() -> QdrantClient:
    global _def_client
    if _def_client is None:
        # Allow opting into gRPC for large upserts; increase timeout.
        prefer_grpc = os.environ.get("QDRANT_PREFER_GRPC", "false").lower() in ("1", "true", "yes")
        grpc_port = int(os.environ.get("QDRANT_GRPC_PORT", "6334"))
        timeout = int(os.environ.get("QDRANT_TIMEOUT", "60"))
        _def_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            timeout=timeout,
        )
    return _def_client


def ensure_collection(name: str, dimension: int):
    qc = get_qdrant()
    existing = [c.name for c in qc.get_collections().collections]
    if name not in existing:
        qc.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )


def ensure_payload_indices(field_names: List[str]):
    """Create payload indices to speed up filtering for A/B tests."""
    qc = get_qdrant()
    for fname in field_names:
        try:
            qc.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=fname,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            # Index may already exist; silently ignore
            pass


def _sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure payload values are JSON-serializable primitives.
    Converts Enums and other objects to strings; preserves dicts/lists recursively.
    """
    def normalize(val: Any) -> Any:
        if val is None or isinstance(val, (str, int, float, bool)):
            return val
        if isinstance(val, enum.Enum):
            return getattr(val, "value", str(val))
        if isinstance(val, (list, tuple)):
            return [normalize(v) for v in val]
        if isinstance(val, dict):
            return {str(k): normalize(v) for k, v in val.items()}
        # Fallback to string representation
        try:
            return str(val)
        except Exception:
            return None

    return {str(k): normalize(v) for k, v in payload.items()}


def upsert_embeddings(vectors: List[Dict[str, Any]]):
    qc = get_qdrant()
    # infer dimension from first vector
    if not vectors:
        return
    dim = len(vectors[0]["values"]) if vectors[0].get("values") else 0
    ensure_collection(QDRANT_COLLECTION, dim)
    # Ensure indices for common filters
    ensure_payload_indices(["chunking.strategy", "run_id"])
    points: List[PointStruct] = []
    for v in vectors:
        pid = v.get("id") or str(uuid4())
        vec = v.get("values")
        if not vec:
            # Skip invalid vectors with no values
            continue
        meta = _sanitize_payload(v.get("metadata", {}))
        points.append(PointStruct(id=pid, vector=vec, payload=meta))
    # Batch upsert to avoid large HTTP payloads on Windows and reduce timeouts
    batch_size = int(os.environ.get("QDRANT_UPSERT_BATCH", "256"))
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        qc.upsert(collection_name=QDRANT_COLLECTION, points=batch, wait=True)


def list_run_ids(limit_per_page: int = 1000, max_pages: int = 100) -> List[str]:
    """Return all distinct run_id values present in the collection."""
    qc = get_qdrant()
    offset = None
    run_ids: set[str] = set()
    for _ in range(max_pages):
        try:
            records, next_offset = qc.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=limit_per_page,
                offset=offset,
                with_payload=True,
            )
        except Exception:
            break
        for r in records:
            payload = getattr(r, "payload", {}) or {}
            rid = payload.get("run_id")
            if rid:
                run_ids.add(str(rid))
        if not next_offset:
            break
        offset = next_offset
    return sorted(run_ids)


def build_filter(flt: Dict[str, Any] | None) -> Filter | None:
    if not flt:
        return None
    conditions = []
    for key, val in flt.items():
        conditions.append(FieldCondition(key=f"{key}", match=MatchValue(value=val)))
    return Filter(must=conditions) if conditions else None


def query_embeddings(embedding: List[float], top_k: int = 5, filter: Dict | None = None):
    qc = get_qdrant()
    flt = build_filter(filter)
    # # Use the high-level client method `search` per qdrant-client docs; avoid direct http API.
    # search = getattr(qc, "search", None)
    # if not callable(search):
    #     raise RuntimeError("Installed qdrant-client does not expose 'search' method on QdrantClient")
    # res = search(
    #     collection_name=QDRANT_COLLECTION,
    #     query_vector=embedding,
    #     limit=top_k,
    #     with_payload=True,
    #     score_threshold=None,
    #     query_filter=flt,
    # )

    res=qc.query_points(
        collection_name=QDRANT_COLLECTION,
        query=embedding,
        query_filter=flt,
        limit=top_k,
        with_payload=True,
    ).points

    # Normalize to a similar shape as Pinecone's matches
    matches = []
    res_iter = cast(Iterable[Any], res)
    for r in res_iter:
        matches.append({
            "id": str(r.id),
            "score": r.score,
            "metadata": r.payload or {},
        })
    return {"matches": matches}

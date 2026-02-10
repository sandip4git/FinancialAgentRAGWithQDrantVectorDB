import os
import sys
from typing import Sequence, Any

# Ensure project root on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from qdrant_client import QdrantClient
from src.config.settings import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION


def _preview_text(text: str, lines: int = 3) -> str:
    if not text:
        return ""
    return "\n".join(text.splitlines()[:lines])


def _preview_vec(vec: Sequence[float] | None, n: int = 8) -> str:
    if not vec:
        return "[]"
    preview = list(vec[:n])
    # Format to 4 decimals for readability
    return "[" + ", ".join(f"{v:.4f}" for v in preview) + "]"


def _extract_vector(record: Any) -> Sequence[float] | None:
    # Qdrant Python client records may expose 'vector' for single-vector collections
    vec = getattr(record, "vector", None)
    if vec is not None:
        return vec
    # For named vectors, 'vectors' might be a dict or a structure with 'values'
    vectors = getattr(record, "vectors", None)
    if hasattr(vectors, "values"):
        return vectors.values
    if isinstance(vectors, dict) and vectors:
        # Take the first named vector
        return next(iter(vectors.values()))
    return None


def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    records, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=5,
        with_payload=True,
        with_vectors=True,
    )
    if not records:
        print("No points found in collection:", QDRANT_COLLECTION)
        return

    print(f"Inspecting first {len(records)} points from '{QDRANT_COLLECTION}':\n")
    for i, r in enumerate(records, 1):
        payload = getattr(r, "payload", {}) or {}
        rid = str(getattr(r, "id", ""))
        src = payload.get("source")
        page = payload.get("page")
        text = payload.get("text") or ""
        vec = _extract_vector(r)
        dim = len(vec) if vec is not None else 0

        print(f"[{i}] id={rid} source={src} page={page} dim={dim}")
        print("Text preview:")
        print(_preview_text(text, lines=3))
        print("Embedding preview (first 8):", _preview_vec(vec, n=8))
        print("-" * 60)


if __name__ == "__main__":
    main()

import os
import sys
import uuid
import logging
import argparse
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Ensure project root on sys.path for src imports
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.rag.embeddings import embed_texts, embedding_dimension
from src.rag.vector_store import upsert_embeddings
from src.rag.chunking import chunk_docs_semantic

DATA_DIR = os.path.join(ROOT, "data", "raw")
LOGGER = logging.getLogger(__name__)


def load_pdfs(path: str) -> List[Dict]:
    docs = []
    for fname in os.listdir(path):
        if not fname.lower().endswith(".pdf"):
            continue
        full = os.path.join(path, fname)
        loader = PyPDFLoader(full)
        docs.extend(loader.load())
    return docs


def chunk_docs(docs, mode: str = "recursive"):
    if mode == "semantic":
        LOGGER.info("Chunking mode: semantic (LangChain SemanticChunker)")
        # Threshold configurable via env, default 0.4
        sem_threshold = float(os.getenv("SEMANTIC_THRESHOLD", "0.4"))
        return chunk_docs_semantic(
            docs,
            breakpoint_threshold_type="percentile",
            threshold=sem_threshold,
        )
    LOGGER.info("Chunking mode: recursive (RecursiveCharacterTextSplitter)")
    chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def build_vectors(chunks, chunking_mode: str) -> List[Dict]:
    texts = [c.page_content for c in chunks]
    LOGGER.debug("Preparing to embed %d chunks", len(texts))
    embeds = embed_texts(texts)
    if embeds:
        LOGGER.debug("Embeddings computed. Dimension=%d", len(embeds[0]))

    # Ingestion run tagging for A/B testing
    run_id = os.getenv("RUN_ID") or str(uuid.uuid4())
    os.environ["RUN_ID"] = run_id  # persist for downstream calls if needed

    # Chunking parameters for metadata
    chunk_meta: Dict[str, Dict] = {}
    if chunking_mode == "semantic":
        chunk_meta = {
            "chunking": {
                "strategy": "semantic",
                "threshold": float(os.getenv("SEMANTIC_THRESHOLD", "0.4")),
            }
        }
    else:
        chunk_meta = {
            "chunking": {
                "strategy": "recursive",
                "chunk_size": int(os.getenv("CHUNK_SIZE", "1200")),
                "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
            }
        }

    out = []
    for i, (c, e) in enumerate(zip(chunks, embeds)):
        meta = c.metadata or {}
        meta.update({
            "source": meta.get("source"),
            "page": meta.get("page"),
            "run_id": run_id,
            "chunk_index": i,
        })
        meta = meta | chunk_meta

        if i < 3:  # show a few samples for insight
            text_preview_lines = "\n".join((c.page_content or "").splitlines()[:3])
            emb_preview = e[:8] if isinstance(e, list) else []
            LOGGER.info(
                "Sample %d -> source=%s page=%s run_id=%s strategy=%s\nText preview:\n%s\nEmbedding preview (first 8): %s",
                i + 1,
                meta.get("source"),
                meta.get("page"),
                run_id,
                meta.get("chunking", {}).get("strategy"),
                text_preview_lines,
                emb_preview,
            )
        out.append({
            "id": str(uuid.uuid4()),
            "values": e,
            "metadata": meta | {"text": c.page_content}
        })
    return out


def main():
    # Basic logging config; set LOG_LEVEL=DEBUG for verbose output
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Ingest quarterly PDFs into Qdrant")
    parser.add_argument(
        "--chunking",
        choices=["recursive", "semantic"],
        default=os.getenv("CHUNKING_MODE", "recursive"),
        help="Chunking strategy",
    )
    args = parser.parse_args()

    if not os.path.isdir(DATA_DIR):
        raise RuntimeError(f"Missing data dir: {DATA_DIR}")
    docs = load_pdfs(DATA_DIR)
    if not docs:
        print("No PDFs found in data/raw. Add quarterly results PDFs and rerun.")
        return
    chunks = chunk_docs(docs, mode=args.chunking)
    vectors = build_vectors(chunks, chunking_mode=args.chunking)
    upsert_embeddings(vectors)
    print(f"Upserted {len(vectors)} vectors to local store (dim={embedding_dimension()}).")


if __name__ == "__main__":
    main()

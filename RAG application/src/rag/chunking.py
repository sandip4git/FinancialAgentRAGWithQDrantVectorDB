from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker, BreakpointThresholdType
import re
import math
from src.rag.embeddings import embed_texts
from langchain_core.embeddings import Embeddings


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def semantic_chunk_text(
    text: str,
    max_chunk_chars: int = 1200,
    min_chunk_chars: int = 400,
    similarity_threshold: float = 0.35,
    overlap_chars: int = 200,
) -> List[str]:
    """
    Chunk text using semantic boundaries with Azure OpenAI embeddings.

    Strategy:
    - Split into paragraphs by blank lines.
    - Compute embeddings per paragraph.
    - Merge adjacent paragraphs while similarity >= threshold and size constraints hold.
    - Start a new chunk on low similarity or when exceeding max size.
    - Include a character overlap between chunks to preserve context.
    """
    text = (text or "").strip()
    if not text:
        return []

    # Paragraph-level segmentation (fallback to full text if single paragraph)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) == 1:
        return chunk_text(text, chunk_size=max_chunk_chars, chunk_overlap=overlap_chars)

    # Embed paragraphs
    vectors = embed_texts(paragraphs)

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return (dot / (na * nb)) if na and nb else 0.0

    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for i, para in enumerate(paragraphs):
        para_len = len(para)
        if not buf:
            buf.append(para)
            buf_len = para_len
            continue

        sim = cosine(vectors[i - 1], vectors[i])
        would_len = buf_len + 1 + para_len  # +1 for a newline join

        if sim >= similarity_threshold and would_len <= max_chunk_chars:
            buf.append(para)
            buf_len = would_len
        else:
            # Flush current buffer as a chunk
            chunk_text_str = "\n\n".join(buf)
            chunks.append(chunk_text_str)
            # Start new buffer with overlap tail from previous chunk
            tail = chunk_text_str[-overlap_chars:] if overlap_chars > 0 else ""
            buf = [tail, para] if tail else [para]
            # Account for potential tail length plus paragraph
            buf_len = len(tail) + para_len

        # If buffer grows too large despite similarity, force flush
        if buf_len >= max_chunk_chars:
            chunk_text_str = "\n\n".join(buf)
            chunks.append(chunk_text_str)
            tail = chunk_text_str[-overlap_chars:] if overlap_chars > 0 else ""
            buf = [tail]
            buf_len = len(tail)

    # Final flush
    if buf_len > 0:
        chunks.append("\n\n".join(buf))

    # Optionally merge very small leading/trailing chunks
    merged: List[str] = []
    for ch in chunks:
        if not merged:
            merged.append(ch)
            continue
        if len(merged[-1]) < min_chunk_chars:
            merged[-1] = (merged[-1] + "\n\n" + ch).strip()
        else:
            merged.append(ch)

    return merged


class AzureLCEmbeddings(Embeddings):
    """Adapter to plug our Azure embedding functions into LangChain's chunkers."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        # Reuse document embedding for queries; caller controls usage
        return embed_texts([text])[0]


def chunk_docs_semantic(
    docs: List,
    breakpoint_threshold_type: BreakpointThresholdType = "percentile",
    threshold: float = 0.4,
):
    """
    Split LangChain Documents using SemanticChunker with Azure embeddings.
    - breakpoint_threshold_type: "percentile" or "stddev"
    - threshold: controls how aggressively to split
    """
    splitter = SemanticChunker(
        embeddings=AzureLCEmbeddings(),
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=threshold,
    )
    return splitter.split_documents(docs)

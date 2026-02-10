import os
import csv
import argparse
from typing import List, Dict

# Ensure project root on sys.path for src imports
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.rag.embeddings import embed_query
from src.rag.vector_store import query_embeddings


def load_prompts(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompts file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def run_compare(prompts: List[str], run_a: str, run_b: str, top_k: int) -> List[Dict]:
    rows: List[Dict] = []
    for prompt in prompts:
        emb = embed_query(prompt)
        res_a = query_embeddings(emb, top_k=top_k, filter={"run_id": run_a})
        res_b = query_embeddings(emb, top_k=top_k, filter={"run_id": run_b})

        matches_a = res_a.get("matches", [])
        matches_b = res_b.get("matches", [])
        ids_a = [m["id"] for m in matches_a]
        ids_b = [m["id"] for m in matches_b]
        # Overlap@k
        overlap = len(set(ids_a) & set(ids_b)) / float(max(len(ids_a), len(ids_b)) or 1)
        # Average score
        avg_score_a = sum(m.get("score", 0.0) for m in matches_a) / float(len(matches_a) or 1)
        avg_score_b = sum(m.get("score", 0.0) for m in matches_b) / float(len(matches_b) or 1)

        rows.append({
            "prompt": prompt,
            "run_a": run_a,
            "run_b": run_b,
            "top_k": top_k,
            "overlap_at_k": overlap,
            "avg_score_a": avg_score_a,
            "avg_score_b": avg_score_b,
            "ids_a": ",".join(ids_a),
            "ids_b": ",".join(ids_b),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare retrieval between two ingested runs by run_id")
    parser.add_argument("--prompts", required=True, help="Path to prompts.txt (one prompt per line)")
    parser.add_argument("--top-k", type=int, default=5, help="Top K for retrieval")
    parser.add_argument("--out", default="apps/eval/results.csv", help="Output CSV path")
    parser.add_argument("--run-a", default=os.getenv("EVAL_RUN_A", ""), help="Run A id (env EVAL_RUN_A)")
    parser.add_argument("--run-b", default=os.getenv("EVAL_RUN_B", ""), help="Run B id (env EVAL_RUN_B)")
    args = parser.parse_args()

    run_a = args.run_a or os.getenv("EVAL_RUN_A")
    run_b = args.run_b or os.getenv("EVAL_RUN_B")
    if not run_a or not run_b:
        raise RuntimeError("Provide run ids via --run-a/--run-b or EVAL_RUN_A/EVAL_RUN_B env vars")

    prompts = load_prompts(args.prompts)
    rows = run_compare(prompts, run_a, run_b, args.top_k)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt",
                "run_a",
                "run_b",
                "top_k",
                "overlap_at_k",
                "avg_score_a",
                "avg_score_b",
                "ids_a",
                "ids_b",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
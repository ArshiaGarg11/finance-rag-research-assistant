"""
retriever.py
------------
Hybrid retrieval over the finance RAG corpus.

Two search methods run in parallel, then their results are merged:
  1. FAISS  — dense vector search (finds semantically similar chunks)
  2. BM25   — keyword search (finds exact term/number matches)

Results are combined using Reciprocal Rank Fusion (RRF), then the
top-k chunks are returned.

Why hybrid?
  - FAISS alone misses exact numbers and tickers ("$4.1B", "LUV")
  - BM25 alone misses paraphrases ("net income" vs "earnings")
  - Together they catch both

Usage:
    retriever = FinanceRetriever()
    retriever.build_index()          # first time — builds + saves indexes
    results = retriever.search("What is Apple's revenue from services?", top_k=5)
    for r in results:
        print(r['score'], r['metadata']['company'], r['text'][:100])
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ── Constants ──────────────────────────────────────────────────────────────────

CHUNKS_DIR   = Path("data/chunks")
INDEX_DIR    = Path("data/index")
EMBED_MODEL  = "all-MiniLM-L6-v2"   # fast, 384-dim, strong for finance text
MIN_WORDS    = 20                     # filter stub chunks shorter than this
RRF_K        = 60                     # reciprocal rank fusion constant


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_all_chunks(min_words: int = MIN_WORDS) -> list[dict]:
    """Load all chunks from data/chunks/, filter tiny stub chunks."""
    all_chunks = []
    for path in sorted(CHUNKS_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            chunks = json.load(f)
        before = len(chunks)
        chunks = [c for c in chunks if len(c["text"].split()) >= min_words]
        print(f"  {path.name}: {before} → {len(chunks)} chunks (filtered {before - len(chunks)} stubs)")
        all_chunks.extend(chunks)
    return all_chunks


def tokenize_for_bm25(text: str) -> list[str]:
    """
    Simple tokenizer for BM25: lowercase, split on whitespace,
    keep alphanumeric tokens and financial symbols ($, %).
    Finance-specific: preserve tokens like '4.1b', 'q1', 'fy2025'
    """
    import re
    # keep letters, digits, $, %, . within tokens
    tokens = re.findall(r"[\$\%]?[a-z0-9][a-z0-9\.\%]*", text.lower())
    return tokens


def reciprocal_rank_fusion(
    faiss_results: list[tuple[int, float]],   # [(chunk_idx, distance), ...]
    bm25_results:  list[tuple[int, float]],   # [(chunk_idx, score), ...]
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    RRF score = 1/(k + rank) for each list, summed.
    Higher is better.
    """
    scores: dict[int, float] = {}

    for rank, (idx, _) in enumerate(faiss_results):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    for rank, (idx, _) in enumerate(bm25_results):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    # sort by combined score descending
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Main class ─────────────────────────────────────────────────────────────────

class FinanceRetriever:
    def __init__(
        self,
        embed_model: str = EMBED_MODEL,
        chunks_dir:  Path = CHUNKS_DIR,
        index_dir:   Path = INDEX_DIR,
    ):
        self.embed_model_name = embed_model
        self.chunks_dir       = chunks_dir
        self.index_dir        = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.chunks:      list[dict]      = []
        self.embed_model: Optional[SentenceTransformer] = None
        self.faiss_index: Optional[faiss.IndexFlatIP]   = None
        self.bm25_index:  Optional[BM25Okapi]           = None

    # ── Index building ─────────────────────────────────────────────────────────

    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Build FAISS and BM25 indexes from all chunks.
        If indexes already exist on disk, load them instead (skip rebuild).
        Set force_rebuild=True to re-index from scratch.
        """
        faiss_path  = self.index_dir / "faiss.index"
        bm25_path   = self.index_dir / "bm25.pkl"
        chunks_path = self.index_dir / "chunks.pkl"

        if not force_rebuild and faiss_path.exists() and bm25_path.exists():
            print("Loading existing indexes from disk...")
            self._load_indexes()
            return

        print("Building indexes from scratch...")

        # ── 1. Load chunks ──────────────────────────────────────────────────
        print("\nLoading chunks:")
        self.chunks = load_all_chunks()
        print(f"\nTotal chunks after filtering: {len(self.chunks)}")

        texts = [c["text"] for c in self.chunks]

        # ── 2. Build FAISS index ────────────────────────────────────────────
        print(f"\nGenerating embeddings with {self.embed_model_name}...")
        self.embed_model = SentenceTransformer(self.embed_model_name)

        # Encode in batches with progress
        embeddings = self.embed_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # for cosine similarity via inner product
        )

        dim = embeddings.shape[1]
        print(f"Embedding shape: {embeddings.shape} (dim={dim})")

        # IndexFlatIP = exact inner product search (= cosine sim when normalized)
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings.astype(np.float32))
        print(f"FAISS index built: {self.faiss_index.ntotal} vectors")

        # ── 3. Build BM25 index ─────────────────────────────────────────────
        print("\nBuilding BM25 index...")
        tokenized = [tokenize_for_bm25(t) for t in texts]
        self.bm25_index = BM25Okapi(tokenized)
        print(f"BM25 index built over {len(tokenized)} documents")

        # ── 4. Save to disk ─────────────────────────────────────────────────
        print("\nSaving indexes to disk...")
        faiss.write_index(self.faiss_index, str(faiss_path))
        with open(bm25_path,  "wb") as f: pickle.dump(self.bm25_index, f)
        with open(chunks_path,"wb") as f: pickle.dump(self.chunks,     f)
        print(f"Saved to {self.index_dir}/")

    def _load_indexes(self) -> None:
        """Load pre-built indexes from disk (fast path)."""
        faiss_path  = self.index_dir / "faiss.index"
        bm25_path   = self.index_dir / "bm25.pkl"
        chunks_path = self.index_dir / "chunks.pkl"

        self.faiss_index = faiss.read_index(str(faiss_path))
        with open(bm25_path,  "rb") as f: self.bm25_index = pickle.load(f)
        with open(chunks_path,"rb") as f: self.chunks      = pickle.load(f)

        print(f"  FAISS: {self.faiss_index.ntotal} vectors")
        print(f"  BM25:  {len(self.chunks)} documents")
        print(f"  Chunks: {len(self.chunks)}")

        # Load embedding model lazily (needed for query encoding)
        print(f"  Loading embedding model ({self.embed_model_name})...")
        self.embed_model = SentenceTransformer(self.embed_model_name)
        print("  Ready.")

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query:        str,
        top_k:        int  = 5,
        faiss_k:      int  = 20,   # candidates from each method before fusion
        bm25_k:       int  = 20,
        company:      str  = None, # optional filter: "JPM", "AAPL", "LUV"
        doc_type:     str  = None, # optional filter: "10-K", "earnings_transcript"
    ) -> list[dict]:
        """
        Hybrid search: FAISS + BM25 → RRF fusion → top_k results.

        Returns list of dicts:
            { "chunk_id", "text", "metadata", "score" }
        sorted by relevance score descending.
        """
        if self.faiss_index is None or self.bm25_index is None:
            raise RuntimeError("Indexes not built. Call build_index() first.")

        # ── 1. FAISS search ─────────────────────────────────────────────────
        query_emb = self.embed_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self.faiss_index.search(query_emb, faiss_k)
        faiss_results = [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]

        # ── 2. BM25 search ──────────────────────────────────────────────────
        query_tokens = tokenize_for_bm25(query)
        bm25_scores  = self.bm25_index.get_scores(query_tokens)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:bm25_k]
        bm25_results = [
            (int(idx), float(bm25_scores[idx]))
            for idx in top_bm25_idx
        ]

        # ── 3. Reciprocal Rank Fusion ───────────────────────────────────────
        fused = reciprocal_rank_fusion(faiss_results, bm25_results)

        # ── 4. Apply metadata filters ───────────────────────────────────────
        results = []
        for idx, rrf_score in fused:
            chunk = self.chunks[idx]
            meta  = chunk["metadata"]

            if company  and meta.get("company")  != company:  continue
            if doc_type and meta.get("doc_type") != doc_type: continue

            results.append({
                "chunk_id": chunk["chunk_id"],
                "text":     chunk["text"],
                "metadata": meta,
                "score":    round(rrf_score, 6),
            })

            if len(results) == top_k:
                break

        return results

    def format_context(self, results: list[dict]) -> str:
        """
        Format retrieved chunks into a context string for the LLM prompt.
        Each chunk is labelled with its source so the model can cite it.
        """
        parts = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            source_label = (
                f"{meta.get('company','?')} "
                f"{meta.get('doc_type','?')} "
                f"{meta.get('year', meta.get('quarter',''))}"
                f" — {meta.get('section', meta.get('speakers',''))}"
            )
            parts.append(f"[Source {i}: {source_label}]\n{r['text']}")
        return "\n\n---\n\n".join(parts)


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    retriever = FinanceRetriever()
    retriever.build_index()

    test_queries = [
        "What is Apple's revenue from services?",
        "How does JPMorgan use interest rate derivatives?",
        "What are Southwest Airlines fuel hedging strategies?",
        "What are the main risk factors for Apple?",
        "What did Jamie Dimon say about AI in the earnings call?",
    ]

    print("\n" + "="*60)
    print("RETRIEVAL TEST")
    print("="*60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.search(query, top_k=3)
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            print(f"  {i}. [{meta.get('company')} {meta.get('doc_type')} "
                  f"{meta.get('year', meta.get('quarter',''))}] "
                  f"score={r['score']:.4f}")
            print(f"     {r['text'][:120]}...")

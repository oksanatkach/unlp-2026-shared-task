"""
Hybrid BM25 + Embedding Retriever with Reciprocal Rank Fusion
Optimized for Ukrainian medical/sports documents.

Dependencies: rank-bm25, sentence-transformers, faiss-cpu, numpy, pymorphy3
"""
import os
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional

from retrievers.preprocessor import UkrainianPreprocessor


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[int, float]]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Tuple[int, float]]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.
    (Cormack, Clarke & Büttcher, SIGIR 2009)

    Formula: RRF(d) = Σ  weight_i / (k + rank_i(d))

    Args:
        ranked_lists: Each list contains (doc_id, score) tuples, best-first.
        k: Smoothing constant. Default 60 (paper-recommended, robust).
           Lower k (20–40) amplifies top ranks; higher k (80+) flattens.
        weights: Per-retriever weights. None = equal weight (standard RRF).

    Returns:
        List of (doc_id, rrf_score) sorted descending.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    rrf_scores: Dict[int, float] = {}
    for weight, results in zip(weights, ranked_lists):
        for rank, (doc_id, _score) in enumerate(results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + weight / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """
    Production-ready hybrid BM25 + embedding retriever with RRF fusion.
    Designed for Ukrainian medical/sports documents.
    """

    def __init__(
        self,
        embedding_model: str,
        use_e5_prefix: bool = True,
        device: str = None,
        custom_preprocessor: Optional[UkrainianPreprocessor] = None,
    ):
        """
        Args:
            embedding_model: HuggingFace model name for sentence-transformers.
            use_e5_prefix: If True, prepend "query: " / "passage: " (required for E5).
            device: "cuda", "cpu", or None (auto-detect).
            custom_preprocessor: Tokenizer for BM25. Defaults to UkrainianPreprocessor.
        """
        self.model = SentenceTransformer(embedding_model, device=device)
        self.use_e5_prefix = use_e5_prefix
        self.dim: int = self.model.get_sentence_embedding_dimension()
        self.preprocess = custom_preprocessor or UkrainianPreprocessor()

        self.documents: List[str] = []
        self.metadata: List[Dict] = []
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.Index] = None

    # ── Indexing ──────────────────────────────

    def index(self, documents: List[str], metadata: List[Dict], batch_size: int = 64):
        """Build both BM25 and FAISS indexes over the document list."""
        self.documents = documents
        self.metadata = metadata

        # BM25: tokenize + lemmatize each document
        tokenized_docs = [self.preprocess(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        # Embeddings: encode all documents
        texts = (
            [f"passage: {doc}" for doc in documents]
            if self.use_e5_prefix
            else documents
        )
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # L2-normalize → cosine via inner product
            convert_to_numpy=True,
        ).astype(np.float32)

        # FAISS: inner product on normalized vectors = cosine similarity
        self.faiss_index = faiss.IndexFlatIP(self.dim)
        self.faiss_index.add(embeddings)

        print(
            f"Indexed {len(documents)} documents | "
            f"BM25 vocab: {len(self.bm25.idf)} terms | "
            f"FAISS: {self.faiss_index.ntotal} vectors × {self.dim}d"
        )

    # ── Individual retrievers ────────────────

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Sparse keyword search. Returns [(doc_idx, score)] descending."""
        tokenized_query = self.preprocess(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    def _embedding_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Dense semantic search. Returns [(doc_idx, score)] descending."""
        q_text = f"query: {query}" if self.use_e5_prefix else query
        q_vec = self.model.encode(
            [q_text], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        distances, indices = self.faiss_index.search(q_vec, top_k)
        return [
            (int(idx), float(dist))
            for idx, dist in zip(indices[0], distances[0])
            if idx >= 0
        ]

    # ── Hybrid search ────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidates_per_retriever: int = 50,
        rrf_k: int = 60,
        weights: Optional[List[float]] = None,
    ) -> List[Dict]:
        """
        Hybrid BM25 + embedding search fused with RRF.

        Args:
            query: Natural-language query in Ukrainian.
            top_k: Number of final results.
            candidates_per_retriever: How many candidates each retriever
                fetches before fusion. Use 3–5× top_k for best results.
            rrf_k: RRF smoothing constant (60 is standard).
            weights: [bm25_weight, embedding_weight]. None = equal.

        Returns:
            List of dicts: {doc_id, text, rrf_score, sources}
        """
        bm25_results = self._bm25_search(query, candidates_per_retriever)
        emb_results = self._embedding_search(query, candidates_per_retriever)

        fused = reciprocal_rank_fusion(
            [bm25_results, emb_results], k=rrf_k, weights=weights
        )

        # Build result set with provenance info
        bm25_set = {doc_id for doc_id, _ in bm25_results}
        emb_set = {doc_id for doc_id, _ in emb_results}

        results = []
        for doc_id, score in fused[:top_k]:
            sources = []
            if doc_id in bm25_set:
                sources.append("bm25")
            if doc_id in emb_set:
                sources.append("embedding")

            result = {
                **self.metadata[doc_id],  # unpacks domain, doc_id, page_number, chunk_id, text
                "rrf_score": round(score, 6),
                "sources": sources,
            }
            results.append(result)

        return results

    # ── Persistence ──────────────────────────

    def save(self, directory: str):
        """Persist both indexes and documents to disk."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.faiss_index, os.path.join(directory, "faiss.index"))
        with open(os.path.join(directory, "bm25.pkl"), "wb") as f:
            pickle.dump(self.bm25, f)
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, directory: str):
        """Load indexes from disk (avoids re-encoding)."""
        self.faiss_index = faiss.read_index(os.path.join(directory, "faiss.index"))
        with open(os.path.join(directory, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)
        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
        with open(os.path.join(directory, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)


if __name__ == "__main__":
    from conf import config
    import csv

    # index_docs_and_save_retriever()

    retriever = HybridRetriever(
        # embedding_model="intfloat/multilingual-e5-base",
        embedding_model=config.embedding_model,
        use_e5_prefix=True,
        device=None
    )
    retriever.load(config.hybrid_retriever_path)
    dev_questions = csv.DictReader(open(config.dev_questions_path))
    # evaluate_retriever(dev_questions, retriever, top_k=15)

    # top 5: doc 0.95; page: 0.84
    # top 10: doc 0.96; page 0.885
    # top 15: doc 0.97; page: 0.91

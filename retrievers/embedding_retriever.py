"""
Pure Embedding Retriever
Semantic-only search using multilingual-e5-base + FAISS.
Drop-in comparable to HybridRetriever for A/B testing.
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional


class EmbeddingRetriever:

    def __init__(
        self,
        embedding_model: str,
        use_e5_prefix: bool = True,
        device: str = None,
    ):
        self.model = SentenceTransformer(embedding_model, device=device)
        self.use_e5_prefix = use_e5_prefix
        self.dim: int = self.model.get_sentence_embedding_dimension()

        self.documents: List[str] = []
        self.metadata: List[dict] = []
        self.faiss_index: Optional[faiss.Index] = None

    # ── Indexing ──────────────────────────────

    def index(self, documents: List[str], metadata: List[dict], batch_size: int = 64):
        self.documents = documents
        self.metadata = metadata

        texts = (
            [f"passage: {doc}" for doc in documents]
            if self.use_e5_prefix
            else documents
        )
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        self.faiss_index = faiss.IndexFlatIP(self.dim)
        self.faiss_index.add(embeddings)

        print(
            f"Indexed {len(documents)} documents | "
            f"FAISS: {self.faiss_index.ntotal} vectors × {self.dim}d"
        )

    # ── Search ───────────────────────────────

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        q_text = f"query: {query}" if self.use_e5_prefix else query
        q_vec = self.model.encode(
            [q_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        distances, indices = self.faiss_index.search(q_vec, top_k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            results.append({
                **self.metadata[idx],
                "score": round(float(score), 6),
            })
        return results

    # ── Persistence ──────────────────────────

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.faiss_index, os.path.join(directory, "faiss.index"))
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, directory: str):
        self.faiss_index = faiss.read_index(os.path.join(directory, "faiss.index"))
        with open(os.path.join(directory, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)
        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)


if __name__ == "__main__":
    from conf import config
    import csv

    # index_docs_and_save_retriever()

    retriever = EmbeddingRetriever(
        embedding_model=config.embedding_model,
        use_e5_prefix=True,
        device=None
    )
    retriever.load(config.semantic_retriever_path)
    dev_questions = csv.DictReader(open(config.dev_questions_path))
    # evaluate_retriever(dev_questions, retriever, top_k=15)

    # top 5: doc 93.9; page 79.4
    # top 10: doc 95.9; page 85
    # top 15: doc 97.4; page 89.2

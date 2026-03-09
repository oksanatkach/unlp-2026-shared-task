from typing import List, Dict
from rank_bm25 import BM25Okapi
import os
import pickle
import numpy as np

from preprocessor import UkrainianPreprocessor


class BM25ChunkRetriever:
    def __init__(self):
        """
        chunks: list of tuples of (chunk id, chunk text)
        """
        self.documents: List[str] = []
        self.metadata: List[dict] = []
        self.preprocess = UkrainianPreprocessor()

    def index(self, documents: List[str], metadata: List[Dict], batch_size: int = 64):
        self.documents = documents
        self.metadata = metadata

        tokenized_docs = [self.preprocess(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        print(
            f"Indexed {len(documents)} documents | "
            f"BM25 vocab: {len(self.bm25.idf)} terms"
        )

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        tokenized_query = self.preprocess(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if scores[idx] == 0:
                continue
            results.append({
                **self.metadata[idx],
                "score": round(float(scores[idx]), 6),
            })

        return results

    def save(self, directory: str):
        """Persist both indexes and documents to disk."""
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "bm25.pkl"), "wb") as f:
            pickle.dump(self.bm25, f)
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, directory: str):
        """Load indexes from disk (avoids re-encoding)."""
        with open(os.path.join(directory, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)
        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
        with open(os.path.join(directory, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)


if __name__ == "__main__":
    import csv
    from conf import config

    # index_docs_and_save_retriever()

    dev_questions_path = config.dev_questions_path
    dev_questions = csv.DictReader(open(dev_questions_path))

    document_retriever = BM25ChunkRetriever()
    document_retriever.load(config.bm25_retriever_path)

    # top 5: doc 91.1; page 82.2
    # top 10: doc 94.8; page 88.5
    # top 15: doc 96.3; page 90.5

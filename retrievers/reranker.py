from sentence_transformers import CrossEncoder
from typing import List, Dict

class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5, batch_size: int = 8) -> List[Dict]:
        pairs = [(query, chunk['text']) for chunk in chunks]
        scores = self.model.predict(pairs, batch_size=batch_size)  # process in smaller batches
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]

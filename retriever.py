from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from typing import List, Dict
import os
import re

tokenizer: AutoTokenizer | None = None

def load_tokenizer():
    global tokenizer
    if tokenizer is None:
        print('Loading tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained("lapa-llm/tokenizer")
        print('Tokenizer loaded')

def tokenize_uk(texts: List[str]) -> list[str]:
    toks = tokenizer(texts, add_special_tokens=False)
    return toks['input_ids']

class BM25ChunkRetriever:
    def __init__(self, chunks: List[Dict[str, str]]):
        """
        chunks: list of tuples of (chunk id, chunk text)
        """
        self.chunks = chunks
        self.corpus_tokens = tokenize_uk([chunk['text'] for chunk in self.chunks])
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def retrieve_chunks(self, question: str, options: list[str] | None = None, top_k: int = 50):
        q = question if not options else (question + " " + "\n".join(options))
        q_tokens = tokenize_uk([q])[0]
        scores = self.bm25.get_scores(q_tokens)  # ndarray-like
        # top-k indices
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{**self.chunks[idx], "score": scores[idx]} for idx in top_idx]

    def retrieve_pages(self,
                       question: str,
                       options: list[str] | None = None,
                       top_k_pages: int = 10,
                       top_k_chunks: int = 100,
                       return_chunk_text: bool = False):
        hits = self.retrieve_chunks(question, options=options, top_k=top_k_chunks)
        # todo: what is this function for?

        # aggregate chunk scores to pages
        # page_best = {}
        # for h in hits:
        #     key = (h['domain'], h["doc_id"], h["page_number"])
        #     current_best_page_chunk = page_best.get(key, None)
        #     if current_best_page_chunk:
        #         current_best_page_chunk_score = current_best_page_chunk[0]
        #         if current_best_page_chunk_score < h['score']:
        #             if return_chunk_text:
        #                 page_best[key] = (h['score'], h['chunk_id'], h['text'])
        #             else:
        #                 page_best[key] = (h['score'], h['chunk_id'])
        #     else:
        #         if return_chunk_text:
        #             page_best[key] = (h['score'], h['chunk_id'], h['text'])
        #         else:
        #             page_best[key] = (h['score'], h['chunk_id'])
        #
        # if return_chunk_text:
        #     top_pages = sorted(
        #         [{"domain": domain, "doc_id": doc_id, "page_number": page_number, "chunk_id": chunk_id, "score": score, "text": text}
        #          for (domain, doc_id, page_number), (score, chunk_id, text) in page_best.items()],
        #         key=lambda x: x["score"],
        #         reverse=True
        #     )[:top_k_pages]
        # else:
        #     top_pages = sorted(
        #         [{"domain": domain, "doc_id": doc_id, "page_number": page_number, "chunk_id": chunk_id, "score": score}
        #          for (domain, doc_id, page_number), (score, chunk_id) in page_best.items()],
        #         key=lambda x: x["score"],
        #         reverse=True
        #     )[:top_k_pages]

        # return top_pages

def prepare_chunks_for_retriever(chunks_path):
    chunks = []

    for root, dirs, files in os.walk(chunks_path):
        for filename in files:
            if filename.endswith('.txt'):
                domain =  re.search(r'\/(domain_.)', root).group(1)
                pdf_id = root.split('/')[-1]
                _, page_number, _, chunk_id = filename.split('.')[0].split('_')
                chunk_text = open(os.path.join(root, filename)).read()
                chunks.append({'domain': domain,
                               'doc_id': pdf_id,
                               'page_number': int(page_number),
                               'chunk_id': int(chunk_id),
                               'text': chunk_text}
                              )

    return chunks


def init_retriever(chunks_path):
    load_tokenizer()
    chunks = prepare_chunks_for_retriever(chunks_path)
    print(f"Processed {len(chunks)} chunks")
    return BM25ChunkRetriever(chunks)


def evaluate_retriever(questions, top_k=5):
    options_columns = ['A', 'B', 'C', 'D', 'E', 'F']
    chunks_path = 'data/output/chunks'
    document_retriever = init_retriever(chunks_path)

    D = 0
    P = 0
    N = 0

    for row in questions:
        question = row['Question']
        options = [row[letter] for letter in options_columns if row[letter]]
        top_chunks = document_retriever.retrieve_chunks(question, options, top_k=top_k)
        values = zip(*[d.values() for d in top_chunks])
        top_chunks = dict(zip(top_chunks[0].keys(), values))

        N += 1
        if row['Domain'] in top_chunks['domain']:
            if row['Doc_ID'].split('.')[0] in top_chunks['doc_id']:
                D += 1
                if int(row['Page_Num']) in top_chunks['page_number']:
                    P += 1

    print(D / N)
    print(P / N)


if __name__ == "__main__":
    import csv

    dev_questions_path = 'data/dev_questions.csv'
    dev_questions = csv.DictReader(open(dev_questions_path))
    evaluate_retriever(dev_questions, top_k=10)

    # top 5: doc 90.5; page 0.75.3
    # top 10: doc 93.5; page 81.8

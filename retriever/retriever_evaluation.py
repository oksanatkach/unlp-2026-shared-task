from collections import defaultdict

def eval_retrieval(questions, options_columns, retrieve_fn, Ks=(1,3,5,10,20)):
    """
    df: dev dataframe
    retrieve_fn: function(row) -> top_pages
        top_pages: list of dicts with keys: doc_id, page_number (int)
    """
    hits_doc = defaultdict(int)
    hits_page = defaultdict(int)
    rr_sum = 0.0
    n = 0

    # proximity stats (only when correct doc found)
    prox_abs_errs = []  # abs(page_pred - page_gold)

    for row in questions:
        gold_doc = row["Doc_ID"].split(".")[0]
        gold_page = int(row["Page_Num"])

        question = row['Question']
        options = [row[letter] for letter in options_columns if row[letter]]
        top_pages = retrieve_fn(question, options)
        n += 1

        # ranks (1-indexed); None if not found
        doc_rank = None
        page_rank = None

        for r, p in enumerate(top_pages, start=1):
            if doc_rank is None and p["doc_id"] == gold_doc:
                doc_rank = r
            if page_rank is None and p["doc_id"] == gold_doc and int(p["page_number"]) == gold_page:
                page_rank = r
            if doc_rank is not None and page_rank is not None:
                break

        # Doc@K and Page@K
        for K in Ks:
            if doc_rank is not None and doc_rank <= K:
                hits_doc[K] += 1
            if page_rank is not None and page_rank <= K:
                hits_page[K] += 1

        # MRR for exact page
        if page_rank is not None:
            rr_sum += 1.0 / page_rank

        # proximity: within correct doc, best page distance among retrieved pages in that doc
        same_doc_pages = [int(p["page_number"]) for p in top_pages if p["doc_id"] == gold_doc]
        if same_doc_pages:
            best_dist = min(abs(p - gold_page) for p in same_doc_pages)
            prox_abs_errs.append(best_dist)

    results = {}
    for K in Ks:
        results[f"Doc@{K}"] = hits_doc[K] / n
        results[f"Page@{K}"] = hits_page[K] / n

    results["MRR_page"] = rr_sum / n
    if prox_abs_errs:
        results["MeanAbsPageError_ifDocFound"] = sum(prox_abs_errs) / len(prox_abs_errs)
        results["MedianAbsPageError_ifDocFound"] = sorted(prox_abs_errs)[len(prox_abs_errs)//2]
        results["DocFoundRate"] = len(prox_abs_errs) / n
    return results


def universal_evaluator(questions, document_retriever, top_k=5):
    options_columns = ['A', 'B', 'C', 'D', 'E', 'F']

    D = 0
    P = 0
    N = 0

    for row in questions:
        question = row['Question']
        options = [row[letter] for letter in options_columns if row[letter]]
        query = question if not options else (question + " " + "\n".join(options))
        top_chunks = document_retriever.search(query, top_k=top_k)
        values = zip(*[d.values() for d in top_chunks])
        top_chunks = dict(zip(top_chunks[0].keys(), values))

        N += 1
        if row['Domain'] in top_chunks['domain']:
            if row['Doc_ID'].split('.')[0] in top_chunks['doc_id']:
                D += 1
                if int(row['Page_Num']) in top_chunks['page_number']:
                    P += 1

    print('Doc:', round((D / N) * 100, 1))
    print('Page:', round((P / N) * 100, 1))


def universal_evaluator_reranker(questions, document_retriever, reranker, reranker_top_k=20, top_k=5):
    options_columns = ['A', 'B', 'C', 'D', 'E', 'F']

    D = 0
    P = 0
    N = 0

    for row in questions:
        question = row['Question']
        options = [row[letter] for letter in options_columns if row[letter]]
        query = question if not options else (question + " " + "\n".join(options))
        top_chunks = document_retriever.search(query, top_k=reranker_top_k)
        top_chunks = reranker.rerank(query, top_chunks, top_k=top_k)

        values = zip(*[d.values() for d in top_chunks])
        top_chunks = dict(zip(top_chunks[0].keys(), values))

        N += 1
        if row['Domain'] in top_chunks['domain']:
            if row['Doc_ID'].split('.')[0] in top_chunks['doc_id']:
                D += 1
                if int(row['Page_Num']) in top_chunks['page_number']:
                    P += 1

    print('Doc:', round((D / N) * 100, 1))
    print('Page:', round((P / N) * 100, 1))

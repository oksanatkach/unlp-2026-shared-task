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


if __name__ == "__main__":
    import retriever
    import csv

    chunks_path = 'data/output/chunks'
    document_retriever = retriever.init_retriever(chunks_path)

    dev_questions_path = 'data/dev_questions.csv'
    dev_questions = csv.DictReader(open(dev_questions_path))
    options_columns = dev_questions.fieldnames[4:10]
    print(eval_retrieval(dev_questions, options_columns, document_retriever.retrieve_pages))


    # row = next(dev_questions)
    # print(row)
    # print(row['Doc_ID'])
    # print(row['Page_Num'])

    # question = row['Question']
    # todo: don't use latin letters for options?
    # currently matching english text and text where ABCDEF appear often
    # todo: replacing BM25 with semantic embeddings should also fix this
    # options = [f"{letter}: {row[letter]}" for letter in options_columns if row[letter]]

    # top_pages, hits = retriever.retrieve_pages(question, options)
    # print(top_pages)
    # print(hits)
    # for el in top_pages:
    #     print(el['domain'])
    #     print(el['doc_id'])
    #     print(el['page_number'])
    #     print(el['score'])
    #     print(el['chunk_id'])
    #     print('#######################################')

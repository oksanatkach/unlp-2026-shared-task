import re, os


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


def index_docs_and_save_retriever(retriever_class, chunks_path, retriever_path, **kwargs):
    chunks = prepare_chunks_for_retriever(chunks_path)
    documents = [chunk['text'] for chunk in chunks]

    retriever = retriever_class(**kwargs)
    retriever.index(documents, chunks, batch_size=32)

    retriever.save(retriever_path)

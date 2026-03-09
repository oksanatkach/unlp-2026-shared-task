import json
import spacy
import os
import shutil

nlp = spacy.load('uk_core_news_sm', disable=["ner", "lemmatizer"])


def get_page_chunks(doc_text, page_ranges, page_number, max_chunk_size=500, overlap=2):
    start, end = page_ranges[page_number]
    page_text = doc_text[start:end]

    prev_page_last_paragraph = ''
    next_page_first_paragraph = ''
    if page_number+1 in page_ranges:
        start, end = page_ranges[page_number+1]
        next_page_text = doc_text[start:end]
        next_page_first_paragraph = next_page_text.strip('\n').split('\n')[0]

    if page_number-1 in page_ranges:
        start, end = page_ranges[page_number-1]
        prev_page_text = doc_text[start:end]
        prev_page_last_paragraph = prev_page_text.strip('\n').split('\n')[-1]

    text = prev_page_last_paragraph + page_text + next_page_first_paragraph

    doc = nlp(text)
    sents = list(doc.sents)

    chunks = []
    chunk = []

    ind = 0

    while ind < len(sents):
        sent = sents[ind]

        if chunk:
            chunk_text = text[chunk[0].start_char: chunk[-1].end_char]
        else:
            chunk_text = ''

        # if within limits, add the next sentence
        if len(chunk_text) + len(sent.text) < max_chunk_size:
            chunk.append(sent)
            ind += 1

        # if the first sentence is over the limit, write it truncated and move on with empty chunk
        elif len(sent.text) > max_chunk_size:
            chunks.append(chunk_text)
            print('WARNING: sentence longer than chunk size, truncating')
            chunks.append(sent.text[:max_chunk_size])
            chunk = []
            ind += 1

        else:
            # if chunk is entirely from the next page: do not write it, stop loop
            if chunk_text in next_page_first_paragraph:
                chunk = []
                break

            # chunk has reached limit, write
            chunks.append(chunk_text)

            # if chunk contained multiple sentences, write previous sentences to chunk
            # do not increment, check combination with current sentence
            if len(chunk) > 1:
                actual_overlap = overlap
                while actual_overlap > 0:
                    overlap_chunk = sents[max(0, ind - actual_overlap):ind]
                    overlap_text = text[overlap_chunk[0].start_char: overlap_chunk[-1].end_char]
                    if len(overlap_text) + len(sent.text) < max_chunk_size:
                        chunk = overlap_chunk
                        break
                    actual_overlap -= 1
                else:
                    # no overlap fits at all, start fresh
                    chunk = []
            else:
                chunk = [sent]
                ind += 1

    if chunk:
        # this should never happen but just in case someone removes next page paragraph
        chunk_text = text[chunk[0].start_char: chunk[-1].end_char]
        if chunk_text not in next_page_first_paragraph:
            chunks.append(chunk_text)

    return chunks

def chunk_page_seamlessly(doc_text, page_ranges, chunks_output_path, pdf_id, page_number, max_chunk_size, overlap):

    chunks = get_page_chunks(doc_text, page_ranges, page_number, max_chunk_size=max_chunk_size, overlap=overlap)

    ind_width = len(str(len(chunks)))

    for ind, chunk in enumerate(chunks):
        with open(f'{chunks_output_path}/{pdf_id}/page_{page_number}_chunk_{ind:0{ind_width}d}.txt', 'w') as outfile:
            outfile.write(chunk)


if __name__ == '__main__':
    # domain = 'domain_1'
    domain = 'domain_2'
    pdf_path = f'../data/{domain}/dev'
    pdf_info_path = f'../data/output/pdf_info/{domain}'
    max_chunk_size = 1000
    sentence_overlap = 2
    chunks_output_path = f'../data/output/chunks_{max_chunk_size}/{domain}'

    for file in os.listdir(pdf_path):
        if file.endswith('.pdf'):
            pdf_id = file.split('.')[0]

            if os.path.exists(f'{chunks_output_path}/{pdf_id}'):
                shutil.rmtree(f'{chunks_output_path}/{pdf_id}')

            os.mkdir(f'{chunks_output_path}/{pdf_id}')

            doc_text = open(f'{pdf_info_path}/{pdf_id}_text.txt', 'r').read()
            page_ranges = json.load(open(f'{pdf_info_path}/{pdf_id}_page_ranges.json'))
            page_ranges = {int(k): v for k, v in page_ranges.items()}

            print(pdf_id)
            for page_number in page_ranges.keys():
                print(page_number)

                chunk_page_seamlessly(doc_text=doc_text,
                                      page_ranges=page_ranges,
                                      chunks_output_path=chunks_output_path,
                                      pdf_id=pdf_id,
                                      page_number=page_number,
                                      max_chunk_size=max_chunk_size,
                                      overlap=sentence_overlap)

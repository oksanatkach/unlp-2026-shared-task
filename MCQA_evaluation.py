import json
from typing import Iterable, Dict

import config
if config.YES_NO_QUESTIONS:
    from question_answering import answer_question_yes_no as answer_question
else:
    from question_answering import answer_question
from question_answering import init as QA_init


def evaluate_pipeline(questions: Iterable[Dict], top_k: int = 5):
    QA_init()

    A = 0
    D = 0
    P = 0
    N = 0
    for row in questions:
        print('Processing a row')
        option_letter, best_chunk = answer_question(row, top_k=top_k)
        print('Finished processing a row')
        N += 1

        if option_letter and row['Correct_Answer'] == option_letter:
            A += 1

        if best_chunk and best_chunk['domain'] == row['Domain']:
            if best_chunk['doc_id'] == row['Doc_ID'].split('.')[0]:
                D += 1

                doc_len = len(json.load(
                    open(f'data/output/pdf_info/{row['Domain']}/{best_chunk['doc_id']}_page_ranges.json')))
                P += 1 - (abs(best_chunk['page_number'] - int(row['Page_Num'])) / doc_len)

    return 0.5 * (A / N) + 0.25 * (D / N) + 0.25 * (P / N)


if __name__ == '__main__':
    import csv

    dev_questions = csv.DictReader(open(config.dev_questions_path))
    score = evaluate_pipeline(dev_questions, config.RETRIEVER_TOP_K)
    print(score)

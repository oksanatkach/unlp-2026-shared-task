import json
from typing import Iterable, Dict
from tqdm import tqdm

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

    questions_list = list(questions)

    with tqdm(total=len(questions_list), desc="Evaluating", unit="row") as pbar:
        for row in questions_list:
            answer = answer_question(row, top_k=top_k)
            N += 1

            if answer:
                option_letter, best_chunk = answer

                if option_letter and row['Correct_Answer'] == option_letter:
                    A += 1

                if best_chunk and best_chunk['domain'] == row['Domain']:
                    if best_chunk['doc_id'] == row['Doc_ID'].split('.')[0]:
                        D += 1

                        doc_len = len(json.load(
                            open(f'{config.pdf_info_path}/{row['Domain']}/{best_chunk['doc_id']}_page_ranges.json')))
                        P += 1 - (abs(best_chunk['page_number'] - int(row['Page_Num'])) / doc_len)

            pbar.update(1)
            pbar.set_postfix({"Answer acc:": f"{A / N:.2%}", "Doc acc:": f"{D / N:.2%}", "Page acc:": f"{P / N:.2%}"})  # live accuracy stats

    final_score = 0.5 * (A / N) + 0.25 * (D / N) + 0.25 * (P / N)
    print(f'Final score: {final_score}')


if __name__ == '__main__':
    import csv

    dev_questions = csv.DictReader(open(config.dev_questions_path))
    evaluate_pipeline(dev_questions, config.RETRIEVER_TOP_K)

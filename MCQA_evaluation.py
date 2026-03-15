import json
from typing import Iterable, Dict
from tqdm import tqdm

from conf import config
from MCQA import question_answering_vllm as QA


def evaluate_pipeline(questions: Iterable[Dict], initial_top_k: int, final_top_k: int):
    QA.init()

    A = 0
    D = 0
    P = 0
    N = 0

    questions_list = list(questions)

    with tqdm(total=len(questions_list), desc="Evaluating", unit="row") as pbar:
        for row in questions_list:
            answer = QA.answer_question_prompt_per_chunk_per_option(row=row,
                                                                    initial_top_k=initial_top_k,
                                                                    final_top_k=final_top_k)
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
    print(f'Final score: {final_score:.2%}')

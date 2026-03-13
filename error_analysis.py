from typing import Iterable, Dict
from tqdm import tqdm

from MCQA.question_answering_vllm import init, answer_question_prompt_per_chunk_per_option

def launch_error_analysis(questions: Iterable[Dict], top_k: int = 5):
    init()

    correct_answer_correct_page = 0
    correct_answer_wrong_page = []
    wrong_answer_correct_page = []
    wrong_answer_wrong_page = []

    questions_list = list(questions)

    with tqdm(total=len(questions_list), desc="Evaluating", unit="row") as pbar:
        for ind, row in enumerate(questions_list):
            correct_answer = False
            correct_page = False

            answer = answer_question_prompt_per_chunk_per_option(row, top_k=top_k)

            if answer:
                option_letter, best_chunk = answer

                if option_letter and row['Correct_Answer'] == option_letter:
                    correct_answer = True

                if best_chunk and best_chunk['domain'] == row['Domain']:
                    if best_chunk['doc_id'] == row['Doc_ID'].split('.')[0]:
                        if best_chunk['page_number'] == int(row['Page_Num']):
                            correct_page = True

                if correct_answer and correct_page:
                    correct_answer_correct_page += 1
                if correct_answer and not correct_page:
                    correct_answer_wrong_page.append(ind)
                if not correct_answer and correct_page:
                    wrong_answer_correct_page.append(ind)
                if not correct_answer and not correct_page:
                    wrong_answer_wrong_page.append(ind)

            pbar.update(1)
            pbar.set_postfix({"correct_answer_correct_page:": f"{correct_answer_correct_page}",
                              "correct_answer_wrong_page:": f"{len(correct_answer_wrong_page)}",
                              "wrong_answer_correct_page:": f"{len(wrong_answer_correct_page)}",
                              "wrong_answer_wrong_page:": f"{len(wrong_answer_wrong_page)}",
                              })

    return (correct_answer_correct_page,
            correct_answer_wrong_page,
            wrong_answer_correct_page,
            wrong_answer_wrong_page)

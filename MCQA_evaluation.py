import json
from typing import Iterable, Dict
from tqdm import tqdm

from conf import config

from MCQA.question_answering import init as QA_init
from MCQA.question_answering import (answer_question_prompt_per_chunk,
                                     answer_question_single_prompt,
                                     answer_question_prompt_per_chunk_per_option)

QA_mode_map = {
    'prompt_per_chunk': answer_question_prompt_per_chunk,
    'single_prompt': answer_question_single_prompt,
    'prompt_per_chunk_per_option': answer_question_prompt_per_chunk_per_option
}


def evaluate_pipeline(questions: Iterable[Dict], top_k: int = 5):
    QA_init()

    A = 0
    D = 0
    P = 0
    N = 0

    questions_list = list(questions)

    with tqdm(total=len(questions_list), desc="Evaluating", unit="row") as pbar:
        for row in questions_list:
            answer = QA_mode_map[config.QA_MODE](row, top_k=top_k)
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


def evaluate_no_retriever(questions: Iterable[Dict]):
    QA_init()

    A = 0
    N = 0

    questions_list = list(questions)

    with tqdm(total=len(questions_list), desc="Evaluating", unit="row") as pbar:
        for row in questions_list:
            option_letter = answer_question_no_retriever(row)
            N += 1

            if option_letter:
                if option_letter and row['Correct_Answer'] == option_letter:
                    A += 1

            pbar.update(1)
            pbar.set_postfix({"Answer acc:": f"{A / N:.2%}"})

    print(f'Final score: {A / N:.2%}')


if __name__ == '__main__':
    import csv
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--qa_mode',
                        type=str,
                        default='REGULAR',
                        choices=['REGULAR', 'YES_NO_QUESTIONS', 'YES_NO_QUESTIONS_DIFF'],
                        help='QA mode to use')
    parser.add_argument('--no_retriever',
                        default=False,
                        action='store_true',
                        help='Disable retriever')
    parser.add_argument('--dev_questions_path',
                        type=str,
                        default=config.dev_questions_path,
                        help='Path to dev questions CSV')
    parser.add_argument('--chunks_path',
                        type=str,
                        default=config.chunks_path,
                        help='Path to retriever chunks')
    parser.add_argument('--pdf_info_path',
                        type=str,
                        default=config.pdf_info_path,
                        help='Path to pdf info files')
    args = parser.parse_args()

    config.QA_MODE = args.qa_mode
    config.chunks_path = args.chunks_path
    config.pdf_info_path = args.pdf_info_path
    dev_questions = csv.DictReader(open(args.dev_questions_path))

    if args.no_retriever:
        evaluate_no_retriever(dev_questions)
    else:
        evaluate_pipeline(dev_questions, config.RETRIEVER_TOP_K)

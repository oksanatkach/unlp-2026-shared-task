from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, List, Tuple
import os

from conf import config
from MCQA import prompt_templates
from retrievers.hybrid_retriever import HybridRetriever

options_columns = ['A', 'B', 'C', 'D', 'E', 'F']
document_retriever: HybridRetriever | None = None
llm : AutoModelForCausalLM | None = None
tokenizer: AutoTokenizer | None = None
option_token_ids: List[str] | None = None
yes_token_ids: List[str] | None = None
no_token_ids: List[str] | None = None


def init():
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    global document_retriever, llm, tokenizer, option_token_ids, yes_token_ids, no_token_ids

    if document_retriever is None:
        document_retriever = HybridRetriever(embedding_model=config.embedding_model_large)
        document_retriever.load(config.hybrid_1000_e5large_retriever_path)

    if llm is None:
        llm = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=config.bnb_config,
            device_map="auto",
            torch_dtype=torch.float32,
        )

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    if option_token_ids is None:
        option_token_ids = tokenizer.convert_tokens_to_ids(['A', '▁A',
                                                            'B', '▁B',
                                                            'C', '▁C',
                                                            'D', '▁D',
                                                            'E', '▁E',
                                                            'F' '▁F',
                                                            ])

    if yes_token_ids is None:
        yes_token_ids = tokenizer.convert_tokens_to_ids(['yes', '▁yes',
                                                         'Yes', '▁Yes',
                                                         'YES', '▁YES'
                                                         ])
        no_token_ids = tokenizer.convert_tokens_to_ids(['no', '▁no',
                                                        'No', '▁No',
                                                        'NO', '▁NO'
                                                        ])


def get_best_answer_from_logits(logits: torch.Tensor, valid_token_ids: List[int]) -> Tuple[int, int]:
    option_mask = torch.zeros_like(logits, dtype=torch.bool)
    option_mask[:, valid_token_ids] = True
    combined_mask = torch.isfinite(logits) & option_mask
    valid_logits = logits[combined_mask]
    if valid_logits.numel() > 0:
        valid_indices = combined_mask.nonzero()
        prompt_ids = valid_indices[:, 0].tolist()
        token_ids = valid_indices[:, 1].tolist()
        sorted_logits = valid_logits.argsort(descending=True).tolist()
        best_element_ind = sorted_logits[0]
        return prompt_ids[best_element_ind], token_ids[best_element_ind]
    return


def get_best_answer_from_logit_diff(logits: torch.Tensor) -> Tuple[int, int]:
    yes_logits = logits[:, yes_token_ids]
    no_logits = logits[:, no_token_ids]
    diff = yes_logits - no_logits

    if diff.numel() > 0:
        highest_diff_id = diff.argmax()
        best_prompt_id = highest_diff_id // diff.shape[1]
        best_token_in_yes_token_ids = highest_diff_id % diff.shape[1]
        best_token_id = yes_token_ids[best_token_in_yes_token_ids]
        return best_prompt_id.item(), best_token_id
    return


def answer_question(row: Dict, top_k: int) -> Tuple[str, Dict]:
    question = row['Question']
    options = [row[letter] for letter in options_columns if row[letter]]
    top_chunks = document_retriever.retrieve_chunks(question, options, top_k=top_k)

    options = zip(options_columns[:len(options)], options)
    options = [': '.join(el) for el in options]
    options = '\n'.join(options)

    prompts = [prompt_templates.prompt_template % (chunk['text'], question, options) for chunk in top_chunks]
    tokens = tokenizer(prompts, return_tensors='pt', padding=True).to("cuda")

    outputs = llm.generate(
        **tokens,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=1
    )
    logits = outputs.scores[0]
    best_answer_result = get_best_answer_from_logits(logits, option_token_ids)

    if best_answer_result:

        best_chunk_id, best_token_id = best_answer_result

        option_letter = tokenizer.convert_ids_to_tokens(best_token_id)
        option_letter = option_letter.strip().strip(':')

        return option_letter, top_chunks[best_chunk_id]


def answer_question_yes_no(row: Dict, top_k: int) -> Tuple[str, Dict]:
    question = row['Question']
    options = [row[letter] for letter in options_columns if row[letter]]
    top_chunks = document_retriever.retrieve_chunks(question, options, top_k=top_k)

    prompts = [
        prompt_templates.prompt_template_yes_no % (chunk_dict["text"], question, row[option_letter])
        for ind, chunk_dict in enumerate(top_chunks)
        for option_letter in options_columns
    ]

    CHUNK_SIZE = 3

    all_logits = []
    for i in range(0, len(prompts), CHUNK_SIZE):
        chunk = prompts[i:i + CHUNK_SIZE]
        tokens = tokenizer(chunk, return_tensors='pt', padding=True).to("cuda")

        with torch.inference_mode():
            outputs = llm.generate(
                **tokens,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=1,
                do_sample=False,
            )

        all_logits.append(outputs.scores[0].cpu())  # [chunk_size, vocab_size]

        del outputs, tokens
        torch.cuda.empty_cache()

    logits = torch.cat(all_logits, dim=0)
    best_answer_result = get_best_answer_from_logits(logits, yes_token_ids)

    if best_answer_result:
        best_prompt_id, best_token_id = best_answer_result

        option_letter = options_columns[best_prompt_id % len(options_columns)]
        best_chunk = top_chunks[best_prompt_id // len(options_columns)]

        return option_letter, best_chunk


def answer_question_yes_no_logit_diff(row: Dict, top_k: int) -> Tuple[str, Dict]:
    question = row['Question']
    options = [row[letter] for letter in options_columns if row[letter]]
    top_chunks = document_retriever.retrieve_chunks(question, options, top_k=top_k)

    prompts = [
        prompt_templates.prompt_template_yes_no % (chunk_dict["text"], question, row[option_letter])
        for ind, chunk_dict in enumerate(top_chunks)
        for option_letter in options_columns
    ]

    CHUNK_SIZE = 3

    all_logits = []
    for i in range(0, len(prompts), CHUNK_SIZE):
        chunk = prompts[i:i + CHUNK_SIZE]
        tokens = tokenizer(chunk, return_tensors='pt', padding=True).to("cuda")

        with torch.inference_mode():
            outputs = llm.generate(
                **tokens,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=1,
                do_sample=False,
            )

        all_logits.append(outputs.scores[0].cpu())  # [chunk_size, vocab_size]

        del outputs, tokens
        torch.cuda.empty_cache()

    logits = torch.cat(all_logits, dim=0)

    best_answer_result = get_best_answer_from_logit_diff(logits)

    if best_answer_result:
        best_prompt_id, best_token_id = best_answer_result

        option_letter = options_columns[best_prompt_id % len(options_columns)]
        best_chunk = top_chunks[best_prompt_id // len(options_columns)]

        return option_letter, best_chunk


def answer_question_no_retriever(row: Dict) -> Tuple[str, Dict]:
    question = row['Question']
    options = [row[letter] for letter in options_columns if row[letter]]
    options = zip(options_columns[:len(options)], options)
    options = [': '.join(el) for el in options]
    options = '\n'.join(options)

    prompt = prompt_templates.prompt_template_no_retriever % (question, options)
    tokens = tokenizer(prompt, return_tensors='pt', padding=True).to("cuda")

    output = llm.generate(
        **tokens,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=1
    )
    logits = output.scores[0]
    best_answer_result = get_best_answer_from_logits(logits, option_token_ids)

    if best_answer_result:

        _, best_token_id = best_answer_result

        option_letter = tokenizer.convert_ids_to_tokens(best_token_id)
        option_letter = option_letter.strip().strip(':')

        return option_letter


if __name__ == '__main__':
    import csv

    init()

    dev_questions = csv.DictReader(open(config.dev_questions_path))
    print(answer_question(next(dev_questions), top_k=config.RETRIEVER_TOP_K))
    # {'Question_ID': '0', 'Domain': 'domain_2', 'N_Pages': '5', 'Question': 'Як рекомендовано приймати ретаболіл дорослим?', 'A': 'внутрішньо', 'B': 'підшкірно', 'C': 'орально', 'D': 'внутрішньовенно', 'E': 'внутрішньом’язово', 'F': 'інгаляційно', 'Correct_Answer': 'E', 'Doc_ID': '4e779acee13fa6e0763fb33d1c83030b8e6ea33d.pdf', 'Page_Num': '1'}

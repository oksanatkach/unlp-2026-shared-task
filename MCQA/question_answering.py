from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Tuple
import os
from collections import Counter

from conf import config
from MCQA import prompt_templates
from retrievers.hybrid_retriever import HybridRetriever

options_columns = ['A', 'B', 'C', 'D', 'E', 'F']
document_retriever: HybridRetriever | None = None
llm : AutoModelForCausalLM | None = None
tokenizer: AutoTokenizer | None = None
option_token_ids: torch.Tensor | None = None
yes_token_id: int | None = None
no_token_id: int | None = None


def init():
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    global document_retriever, llm, tokenizer, option_token_ids, yes_token_id, no_token_id

    if document_retriever is None:
        document_retriever = HybridRetriever(embedding_model=config.embedding_model_large)
        document_retriever.load(config.hybrid_1000_e5large_retriever_path)

    if llm is None:
        llm = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=config.bnb_config,
            device_map="auto",
            dtype=torch.float32,
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
        option_token_ids = torch.tensor(option_token_ids, device='cpu')

    if yes_token_id is None:
        yes_token_id = tokenizer.convert_tokens_to_ids('так')

    if no_token_id is None:
        no_token_id = tokenizer.convert_tokens_to_ids('ні')


def answer_question_prompt_per_chunk(row: Dict, top_k: int) -> Tuple[str, Dict]:
    question = row['Question']
    options = [row[letter] for letter in options_columns if row[letter]]
    query = question + " " + "\n".join(options)
    top_chunks = document_retriever.search(query, top_k=top_k)

    options_labeled = '\n'.join([': '.join(el) for el in zip(options_columns[:len(options)], options)])

    # this is to make sure the next token logit will be option letter, not a space token
    if not prompt_templates.prompt_template.endswith(' '):
        prompt_templates.prompt_template += ' '

    prompts = [prompt_templates.prompt_template % (chunk['text'], question, options_labeled) for chunk in top_chunks]

    tokens = tokenizer(prompts, return_tensors='pt', padding=True).to("cuda")

    with torch.no_grad():
        outputs = llm(**tokens)

    next_token_logits = outputs.logits[:, -1, :].to("cpu")
    selected_logits = next_token_logits[:, option_token_ids]
    best_option_idx = selected_logits.argmax(dim=-1)
    best_token_ids = option_token_ids[best_option_idx]
    best_logits = selected_logits[torch.arange(selected_logits.shape[0]), best_option_idx]
    option_letters = [token.strip() for token in tokenizer.convert_ids_to_tokens(best_token_ids)]

    option_counter = Counter(option_letters)
    max_count = max(option_counter.values())
    top_letters = [l for l, c in option_counter.items() if c == max_count]
    answer_letter = (
        top_letters[0] if len(top_letters) == 1
        else max(top_letters, key=lambda l: sum(
            best_logits[i].item() for i, ol in enumerate(option_letters) if ol == l
        ))
    )

    winning_indices = [i for i, ol in enumerate(option_letters) if ol == answer_letter]
    best_chunk = top_chunks[min(winning_indices)]

    return answer_letter, best_chunk


def answer_question_single_prompt(row: Dict, top_k: int) -> Tuple[str, Dict]:
    question = row['Question']
    options = [row[letter] for letter in options_columns if row[letter]]
    query = question + " " + "\n".join(options)
    top_chunks = document_retriever.search(query, top_k=top_k)
    all_chunks_text = '\n\n'.join([chunk['text'] for chunk in top_chunks])

    options_labeled = '\n'.join([': '.join(el) for el in zip(options_columns[:len(options)], options)])

    # this is to make sure the next token logit will be option letter, not a space token
    if not prompt_templates.prompt_template.endswith(' '):
        prompt_templates.prompt_template += ' '
    prompt = prompt_templates.prompt_template % (all_chunks_text, question, options_labeled)

    tokens = tokenizer(prompt, return_tensors='pt', padding=True).to("cuda")

    with torch.no_grad():
        outputs = llm(**tokens)

    next_token_logits = outputs.logits[:, -1, :].to("cpu")
    selected_logits = next_token_logits[:, option_token_ids]
    best_option_idx = selected_logits.argmax(dim=-1)
    best_token_ids = option_token_ids[best_option_idx]
    answer_letter = tokenizer.convert_ids_to_tokens(best_token_ids)[0].strip()

    return answer_letter, top_chunks[0]


def answer_question_prompt_per_chunk_per_option(row: Dict, top_k: int) -> Tuple[str, Dict]:
    question = row['Question']
    options = [row[letter] for letter in options_columns if row[letter]]
    query = question + " " + "\n".join(options)
    top_chunks = document_retriever.search(query, top_k=top_k)

    options_labeled = [': '.join(el) for el in zip(options_columns[:len(options)], options)]

    # this is to make sure the next token logit will be option letter, not a space token
    if not prompt_templates.prompt_template_yes_no.endswith(' '):
        prompt_templates.prompt_template_yes_no += ' '

    option_scores = []
    option_chunk_margins = []  # per-option, per-chunk margins

    for option in options_labeled:
        prompts = [prompt_templates.prompt_template_yes_no % (chunk['text'], question, option) for chunk in top_chunks]
        tokens = tokenizer(prompts, return_tensors='pt', padding=True).to("cuda")

        with torch.no_grad():
            outputs = llm(**tokens)

        next_token_logits = outputs.logits[:, -1, :]
        yes_logits = next_token_logits[:, yes_token_id]
        no_logits = next_token_logits[:, no_token_id]
        margins = yes_logits - no_logits  # (top_k,)

        option_scores.append(margins.max().item())
        option_chunk_margins.append(margins)  # store full tensor

    best_option_idx = max(range(len(options)), key=lambda i: option_scores[i])
    answer_letter = options_columns[best_option_idx]

    # now use the margins for the winning option to find best chunk
    best_chunk_idx = option_chunk_margins[best_option_idx].argmax().item()
    best_chunk = top_chunks[best_chunk_idx]

    return answer_letter, best_chunk

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


def init():
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    global document_retriever, llm, tokenizer, option_token_ids

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


def answer_question(row: Dict, top_k: int) -> Tuple[str, Dict]:
    question = row['Question']
    options = [row[letter] for letter in options_columns if row[letter]]
    query = question + " " + "\n".join(options)
    top_chunks = document_retriever.search(query, top_k=top_k)

    options_labeled = '\n'.join([': '.join(el) for el in zip(options_columns[:len(options)], options)])

    # ── Step 1: get answer via voting (your current method) ──
    prompts = [prompt_templates.prompt_template % (chunk['text'], question, options_labeled) for chunk in top_chunks]

    # also add a no-context prompt as baseline
    prompts_with_baseline = prompts + [prompt_templates.no_context_template % (question, options_labeled)]

    tokens = tokenizer(prompts_with_baseline, return_tensors='pt', padding=True).to("cuda")
    with torch.no_grad():
        outputs = llm(**tokens)

    next_token_logits = outputs.logits[:, -1, :]
    selected_logits = next_token_logits[:, option_token_ids]
    best_option_idx = selected_logits.argmax(dim=-1)
    best_token_ids = option_token_ids[best_option_idx]
    best_logits = selected_logits[torch.arange(selected_logits.shape[0]), best_option_idx]
    option_letters = [token.strip() for token in tokenizer.convert_ids_to_tokens(best_token_ids)]

    # voting on chunks only (exclude baseline at index -1)
    chunk_letters = option_letters[:-1]
    chunk_logits = best_logits[:-1]
    baseline_logits = selected_logits[-1]  # full distribution for baseline

    option_counter = Counter(chunk_letters)
    max_count = max(option_counter.values())
    top_letters = [l for l, c in option_counter.items() if c == max_count]
    answer_letter = (
        top_letters[0] if len(top_letters) == 1
        else max(top_letters, key=lambda l: sum(
            chunk_logits[i].item() for i, ol in enumerate(chunk_letters) if ol == l
        ))
    )

    # ── Step 2: attribute page by logit lift over baseline ──
    winning_letter_token = option_token_ids[
        [tokenizer.convert_ids_to_tokens(t.item()).strip() for t in option_token_ids].index(answer_letter)
    ]
    baseline_logit_for_answer = baseline_logits[
        (option_token_ids == winning_letter_token).nonzero()[0]
    ].item()

    # chunk that most increases confidence in the answer vs baseline
    best_chunk_idx = max(
        range(len(top_chunks)),
        key=lambda i: selected_logits[i][
                          (option_token_ids == winning_letter_token).nonzero()[0]
                      ].item() - baseline_logit_for_answer
    )
    best_chunk = top_chunks[best_chunk_idx]

    return answer_letter, best_chunk
git
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


def answer_question(row: Dict, top_k: int) -> Tuple[str | None, Dict | None]:
    question = row['Question']
    options = [row[letter] for letter in options_columns if row[letter]]
    query = question if not options else (question + " " + "\n".join(options))
    top_chunks = document_retriever.search(query, top_k=top_k)

    options = zip(options_columns[:len(options)], options)
    options = [': '.join(el) for el in options]
    options = '\n'.join(options)

    prompts = [prompt_templates.prompt_template % (chunk['text'], question, options) for chunk in top_chunks]
    tokens = tokenizer(prompts, return_tensors='pt', padding=True).to("cuda")

    # no need to call generate -- just do a single forward pass
    with torch.no_grad():
        outputs = llm(**tokens)

    next_token_logits = outputs.logits[:, -1, :].to("cpu")

    # get only option letters' logits
    selected_logits = next_token_logits[:, option_token_ids]
    best_option_idx = selected_logits.argmax(dim=-1)
    # get best option letter per row
    best_token_ids = option_token_ids[best_option_idx]
    # get best logit per row
    best_logits = selected_logits[torch.arange(selected_logits.shape[0]), best_option_idx]

    option_letters = [token.strip() for token in tokenizer.convert_ids_to_tokens(best_token_ids)]
    option_counter = Counter(option_letters)
    max_count = max(option_counter.values())
    top_letters = [l for l, c in option_counter.items() if c == max_count]

    if len(top_letters) == 1:
        answer_letter = top_letters[0]
    else:
        # break tie by sum of logits for each tied letter
        answer_letter = max(
            top_letters,
            key=lambda l: sum(best_logits[i].item() for i, ol in enumerate(option_letters) if ol == l)
        )

    # best chunk: highest logit among chunks that voted for the winning letter
    winning_indices = [i for i, ol in enumerate(option_letters) if ol == answer_letter]
    best_chunk_idx = max(winning_indices, key=lambda i: best_logits[i].item())
    best_chunk = top_chunks[best_chunk_idx]

    return answer_letter, best_chunk

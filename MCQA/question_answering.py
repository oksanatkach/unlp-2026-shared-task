from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, List, Tuple, Option
import os

from conf import config
from MCQA import prompt_templates
from retrievers.hybrid_retriever import HybridRetriever

options_columns = ['A', 'B', 'C', 'D', 'E', 'F']
document_retriever: HybridRetriever | None = None
llm : AutoModelForCausalLM | None = None
tokenizer: AutoTokenizer | None = None
option_token_ids: List[str] | None = None


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


def get_best_answer_from_logits(logits: torch.Tensor, valid_token_ids: List[int])\
        -> Tuple[Option[int, None], Option[int, None]]:
    option_mask = torch.zeros_like(logits, dtype=torch.bool)
    option_mask[:, valid_token_ids] = True
    valid_logits = logits[option_mask]
    if valid_logits.numel() > 0:
        valid_indices = option_mask.nonzero()
        token_ids = valid_indices[:, 1].tolist()
        sorted_logits_ids = valid_logits.argsort(descending=True).tolist()
        best_element_ind = sorted_logits_ids[0]
        return token_ids[best_element_ind], valid_logits[best_element_ind].item()
    return None, None


def answer_question(row: Dict, top_k: int) -> Tuple[Option[str, None], Option[Dict, None]]:
    question = row['Question']
    options = [row[letter] for letter in options_columns if row[letter]]
    query = question if not options else (question + " " + "\n".join(options))
    top_chunks = document_retriever.search(query, top_k=top_k)

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

    # Step 1: for each chunk, independently get the best answer letter + its logit
    votes = {}  # letter -> list of logits
    chunk_votes = []  # per-chunk: (letter, logit)

    for chunk_idx in range(logits.shape[0]):
        chunk_logits = logits[chunk_idx].unsqueeze(0)
        token_id, token_logit = get_best_answer_from_logits(chunk_logits, option_token_ids)
        if token_id:
            letter = tokenizer.convert_ids_to_tokens(token_id).strip()
            chunk_votes.append((letter, token_logit, top_chunks[chunk_idx]))
            votes.setdefault(letter, []).append(token_logit)

    if not chunk_votes:
        return None, None

    # Step 2: pick answer by majority vote, break ties by sum of logits
    best_letter = max(votes, key=lambda l: (len(votes[l]), sum(votes[l])))

    # Step 3: attribute page to the chunk that voted for the winning answer with highest confidence
    winning_votes = [(logit, chunk) for letter, logit, chunk in chunk_votes if letter == best_letter]
    best_logit, best_chunk = max(winning_votes, key=lambda x: x[0])

    return best_letter, best_chunk

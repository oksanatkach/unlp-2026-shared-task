from transformers import Gemma3ForCausalLM, AutoTokenizer
from typing import Dict, Tuple
import torch
import os

from conf import config
from MCQA import prompt_templates
from retriever.hybrid_retriever import HybridRetriever
from retriever.reranker import CrossEncoderReranker

options_columns = ['A', 'B', 'C', 'D', 'E', 'F']
document_retriever: HybridRetriever | None = None
llm: Gemma3ForCausalLM | None = None
tokenizer: AutoTokenizer | None = None
reranker: CrossEncoderReranker | None = None
yes_token_id: int | None = None
no_token_id: int | None = None


def init():
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    global document_retriever, llm, tokenizer, reranker, yes_token_id, no_token_id

    if llm is None:
        llm = Gemma3ForCausalLM.from_pretrained(
            config.llm_model_name,
            quantization_config=config.bnb_config,
            device_map="auto",
            max_memory={0: "10GiB", 1: "10GiB", "cpu": "20GiB"},
        )

    if document_retriever is None:
        document_retriever = HybridRetriever(embedding_model=config.embedding_model_name, device="cpu")
        document_retriever.load(config.retriever_path)

    if reranker is None:
        reranker = CrossEncoderReranker(model_name=config.reranker_model_name, device="cpu")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name)
        yes_token_id = tokenizer.convert_tokens_to_ids('так')
        no_token_id = tokenizer.convert_tokens_to_ids('ні')


def answer_question_prompt_per_chunk_per_option(row: Dict, retriever_top_k: int, reranker_top_k: int) -> Tuple[str, Dict]:
    question = row['Question']
    options = [row[letter] for letter in options_columns if row[letter]]
    query = question + " " + "\n".join(options)
    top_chunks = document_retriever.search(query, top_k=retriever_top_k)
    top_chunks = reranker.rerank(query, top_chunks, top_k=reranker_top_k)

    # this is to make sure the next token logit will be option letter, not a space token
    if not prompt_templates.prompt_template_yes_no.endswith(' '):
        prompt_templates.prompt_template_yes_no += ' '

    option_scores = []
    option_chunk_margins = []  # per-option, per-chunk margins

    for option in options:

        prompts = [prompt_templates.prompt_template_yes_no % (chunk['text'], question, option) for chunk in top_chunks]
        tokens = tokenizer(prompts, return_tensors='pt', padding=True).to("cuda")

        with torch.no_grad():
            outputs = llm(**tokens)

        next_token_logits = outputs.logits[:, -1, :].to("cpu")
        del outputs
        yes_logits = next_token_logits[:, yes_token_id]
        no_logits = next_token_logits[:, no_token_id]
        margins = yes_logits - no_logits  # (top_k,)

        option_scores.append(margins.max().item())
        option_chunk_margins.append(margins)  # store full tensor

    best_option_idx = max(range(len(options)), key=lambda i: option_scores[i])
    answer_letter = options_columns[best_option_idx]

    ###################
    # now use the margins for the winning option to find best chunk
    positive_indices = [i for i in range(min(3, len(top_chunks)))
                        if option_chunk_margins[best_option_idx][i].item() > 0]

    best_chunk_idx = min(positive_indices) if positive_indices else 0

    best_chunk = top_chunks[best_chunk_idx]

    return answer_letter, best_chunk

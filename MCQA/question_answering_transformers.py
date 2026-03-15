from typing import Dict, Tuple
import torch

from MCQA import prompt_templates
from MCQA.objects import llm, document_retriever, tokenizer, reranker
from MCQA.objects import options_columns, yes_token_id, no_token_id


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

        with torch.inference_mode():
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

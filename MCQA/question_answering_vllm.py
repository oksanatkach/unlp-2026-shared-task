from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import Dict, Tuple
import torch
import os

from conf import config
from MCQA import prompt_templates
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.reranker import CrossEncoderReranker

options_columns = ['A', 'B', 'C', 'D', 'E', 'F']
document_retriever: HybridRetriever | None = None
llm: LLM | None = None
tokenizer: AutoTokenizer | None = None
reranker: CrossEncoderReranker | None = None
yes_token_id: int | None = None
no_token_id: int | None = None
sampling_params = SamplingParams(max_tokens=1,
                                 temperature=0,
                                 logprobs=10,
                                 skip_special_tokens=True)


def init():
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    global document_retriever, llm, tokenizer, reranker, yes_token_id, no_token_id

    if document_retriever is None:
        document_retriever = HybridRetriever(embedding_model=config.embedding_model_large)
        document_retriever.load(config.hybrid_1000_e5large_retriever_path)

    if llm is None:
        llm = LLM(
            model=config.model_name,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enforce_eager=True,
            trust_remote_code=True,
            mm_processor_kwargs={"use_fast": True},
        )

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        yes_token_id = tokenizer.convert_tokens_to_ids('так')
        no_token_id = tokenizer.convert_tokens_to_ids('ні')

    if reranker is None:
        reranker = CrossEncoderReranker(model_name=config.reranker_model_name)


def get_logprob(logprobs_dict: dict, token_id: int) -> float:
    """
    Retrieve logprob for a token_id from vLLM's top-k logprobs dict.
    Falls back to the minimum value in the dict if the token is not in the top-k.
    The minimum value is further discounted by 0.3 to heuristically approach the true missing logprob.
    Note: If the prompt is working correctly it should never be the case that the expected token is missing
    in the top N logprobs.

    Args:
        logprobs_dict: output.outputs[0].logprobs[0]  (dict[int, Logprob])
        token_id: target token ID
    """
    if token_id in logprobs_dict:
        return logprobs_dict[token_id].logprob

    min_logprob_in_top_N = min(v.logprob for v in logprobs_dict.values())
    return min_logprob_in_top_N * 1.3


def answer_question_prompt_per_chunk_per_option(row: Dict, initial_top_k: int, final_top_k: int) -> Tuple[str, Dict]:
    question = row['Question']
    options = [row[letter] for letter in options_columns if row[letter]]
    query = question + " " + "\n".join(options)
    top_chunks = document_retriever.search(query, top_k=initial_top_k)
    top_chunks = reranker.rerank(query, top_chunks, top_k=final_top_k)

    # this is to make sure the next token logit will be option letter, not a space token
    if not prompt_templates.prompt_template_yes_no.endswith(' '):
        prompt_templates.prompt_template_yes_no += ' '

    option_scores = []
    option_chunk_margins = []  # per-option, per-chunk margins

    for option in options:

        prompts = [prompt_templates.prompt_template_yes_no % (chunk['text'], question, option) for chunk in top_chunks]
        formatted = tokenizer.apply_chat_template(
            [[{"role": "user", "content": prompt}] for prompt in prompts],
            tokenize=False,
            add_generation_prompt=True
        )
        outputs = llm.generate(formatted, sampling_params, use_tqdm=False)

        yes_logprobs = torch.tensor([get_logprob(output.outputs[0].logprobs[0], yes_token_id) for output in outputs])
        no_logprobs = torch.tensor([get_logprob(output.outputs[0].logprobs[0], no_token_id) for output in outputs])

        margins = yes_logprobs - no_logprobs  # (top_k,)
        option_scores.append(margins.max().item())
        option_chunk_margins.append(margins)  # store full tensor

    best_option_idx = max(range(len(options)), key=lambda i: option_scores[i])
    answer_letter = options_columns[best_option_idx]

    ###################
    # now use the margins for the winning option to find best chunk
    positive_indices = [i for i in range(len(top_chunks))
                        if option_chunk_margins[best_option_idx][i].item() > 0]

    best_chunk_idx = min(positive_indices) if positive_indices else 0

    best_chunk = top_chunks[best_chunk_idx]

    return answer_letter, best_chunk

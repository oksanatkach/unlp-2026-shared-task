from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
from typing import Dict, Tuple
import os

from conf import config
from MCQA import prompt_templates
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.reranker import CrossEncoderReranker
from MCQA.utils import clear_dir, read_dir_logits

options_columns = ['A', 'B', 'C', 'D', 'E', 'F']
document_retriever: HybridRetriever | None = None
llm: LLM | None = None
tokenizer: AutoTokenizer | None = None
reranker: CrossEncoderReranker | None = None
option_token_ids: torch.Tensor | None = None
yes_token_id: int | None = None
no_token_id: int | None = None


def init():
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    global document_retriever, llm, tokenizer, reranker, option_token_ids, yes_token_id, no_token_id

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

    if reranker is None:
        reranker = CrossEncoderReranker(model_name=config.reranker_model_name)

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


def answer_question_prompt_per_chunk_per_option(row: Dict, top_k: int) -> Tuple[str, Dict]:
    question = row['Question']
    options = [row[letter] for letter in options_columns if row[letter]]
    query = question + " " + "\n".join(options)
    top_chunks = document_retriever.search(query, top_k=20)
    top_chunks = reranker.rerank(query, top_chunks, top_k=top_k)

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
        params_list = [
            SamplingParams(seed=idx, max_tokens=1, temperature=0, skip_special_tokens=True)
            for idx in range(top_k)
        ]

        clear_dir(config.tmp_vllm_path)

        llm.generate(formatted, params_list, use_tqdm=False)

        captured = read_dir_logits(config.tmp_vllm_path, top_k)

        yes_logits = captured[:, 0]
        no_logits = captured[:, 1]
        margins = yes_logits - no_logits  # (top_k,)

        option_scores.append(margins.max().item())

        option_chunk_margins.append(margins)  # store full tensor

    best_option_idx = max(range(len(options)), key=lambda i: option_scores[i])
    answer_letter = options_columns[best_option_idx]

    ###################
    # now use the margins for the winning option to find best chunk
    POSITIVE_THRESHOLD = 1.0
    positive_indices = [i for i in range(len(top_chunks))
                        if option_chunk_margins[best_option_idx][i].item() > POSITIVE_THRESHOLD]

    best_chunk_idx = min(positive_indices) if positive_indices else 0

    best_chunk = top_chunks[best_chunk_idx]

    return answer_letter, best_chunk


def answer_question_prompt_per_chunk_per_option_english(row: Dict, top_k: int) -> Tuple[str, Dict]:
    question = row['Question']
    options = [row[letter] for letter in options_columns if row[letter]]
    query = question + " " + "\n".join(options)
    top_chunks = document_retriever.search(query, top_k=top_k)

    # this is to make sure the next token logit will be option letter, not a space token
    if not prompt_templates.prompt_template_yes_no_english.endswith(' '):
        prompt_templates.prompt_template_yes_no_english += ' '

    option_scores = []
    option_chunk_margins = []  # per-option, per-chunk margins

    for option in options:

        prompts = [prompt_templates.prompt_template_yes_no_english % (chunk['text'], question, option) for chunk in top_chunks]
        formatted = tokenizer.apply_chat_template(
            [[{"role": "user", "content": prompt}] for prompt in prompts],
            tokenize=False,
            add_generation_prompt=True
        )
        params_list = [
            SamplingParams(seed=idx, max_tokens=1, temperature=0, skip_special_tokens=True)
            for idx in range(top_k)
        ]

        clear_dir(config.tmp_vllm_path)

        llm.generate(formatted, params_list, use_tqdm=False)

        captured = read_dir_logits(config.tmp_vllm_path, top_k)

        yes_logits = captured[:, 0]
        no_logits = captured[:, 1]
        margins = yes_logits - no_logits  # (top_k,)

        option_scores.append(margins.max().item())
        option_chunk_margins.append(margins)  # store full tensor

    best_option_idx = max(range(len(options)), key=lambda i: option_scores[i])
    answer_letter = options_columns[best_option_idx]

    ###################
    # now use the margins for the winning option to find best chunk
    POSITIVE_THRESHOLD = 1.0
    positive_indices = [i for i in range(len(top_chunks))
                        if option_chunk_margins[best_option_idx][i].item() > POSITIVE_THRESHOLD]

    best_chunk_idx = min(positive_indices) if positive_indices else 0

    best_chunk = top_chunks[best_chunk_idx]

    return answer_letter, best_chunk

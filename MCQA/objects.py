import os
import logging
import torch
from transformers import AutoTokenizer

from conf import config
from MCQA.device import device, load_method
from retriever.hybrid_retriever import HybridRetriever
from retriever.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

if load_method == 'vllm_server':
    llm = None
elif load_method == 'VLLM':
    from vllm import LLM, SamplingParams

    llm: LLM | None = None
    sampling_params = SamplingParams(max_tokens=1,
                                     temperature=0,
                                     logprobs=10,
                                     skip_special_tokens=True)
else:
    from transformers import AutoModelForCausalLM
    llm: AutoModelForCausalLM | None = None

options_columns = ['A', 'B', 'C', 'D', 'E', 'F']
document_retriever: HybridRetriever | None = None
reranker: CrossEncoderReranker | None = None
tokenizer: AutoTokenizer | None = None
yes_token_id: int | None = None
no_token_id: int | None = None


def load_llm():
    if load_method == 'VLLM' and 'A100' in device:
        # use bfloat16
        llm = LLM(
            model=config.llm_model_name,
            dtype="bfloat16",
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enforce_eager=True,
            trust_remote_code=True,
            mm_processor_kwargs={"use_fast": True},
        )
        return llm

    elif load_method == 'VLLM':
        from transformers import AutoConfig
        from vllm.config import AttentionConfig
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        llm_config = AutoConfig.from_pretrained(config.llm_model_name)
        text_config_dict = llm_config.text_config.to_dict()

        hf_overrides = {
            **text_config_dict,
            # do NOT override architectures — keep Gemma3ForConditionalGeneration
            # so weight keys like 'language_model.*' still match
        }
        # remove the None architectures key from text_config
        hf_overrides.pop('architectures', None)

        llm = LLM(
            model=config.llm_model_name,
            dtype="float16",
            tensor_parallel_size=torch.cuda.device_count(),
            enforce_eager=True,
            hf_overrides=hf_overrides,
            max_model_len=2048,
            gpu_memory_utilization=0.80,
            limit_mm_per_prompt={"image": 0},
            attention_config=AttentionConfig(backend=AttentionBackendEnum.TRITON_ATTN)
        )
        return llm

    else:
        llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )

        return llm


def load_retriever(device):
    document_retriever = HybridRetriever(embedding_model=config.embedding_model_name, device=device)
    document_retriever.load(config.retriever_path)
    return document_retriever


def init():
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    global llm, document_retriever, reranker, tokenizer, yes_token_id, no_token_id

    if load_method != "vllm_server" and llm is None:
        llm = load_llm()

    if document_retriever is None:
        document_retriever = load_retriever(device=config.retriever_device)

    if reranker is None:
        reranker = CrossEncoderReranker(model_name=config.reranker_model_name, device=config.retriever_device)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name)
        yes_token_id = tokenizer.convert_tokens_to_ids('так')
        no_token_id = tokenizer.convert_tokens_to_ids('ні')

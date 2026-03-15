from transformers import BitsAndBytesConfig
import torch
import os


USE_VLLM = os.getenv('USE_VLLM') or False
dev_questions_path = os.getenv('DEV_QUESTIONS_PATH') or '../data/dev_questions.csv'
chunks_path = os.getenv('CHUNKS_PATH') or '../data/output/chunks_1000'
pdf_info_path = os.getenv('PDF_INFO_PATH') or '../data/output/pdf_info'
retriever_path = os.getenv('RETRIEVER_PATH') or '../data/uk_hybrid_index_1000_e5large'
embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME') or 'intfloat/multilingual-e5-large'
reranker_model_name = os.getenv('RERANKER_MODEL_NAME') or 'BAAI/bge-reranker-v2-m3'
llm_model_name = os.getenv('LLM_MODEL_NAME') or 'lapa-llm/lapa-v0.1.2-instruct'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,  # ← this is the key change
)

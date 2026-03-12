from transformers import BitsAndBytesConfig
import torch
import os


QA_MODE = os.getenv('QA_MODE') or 'REGULAR'
USE_VLLM = os.getenv('USE_VLLM') or False
RETRIEVER_TOP_K = os.getenv('RETRIEVER_TOP_K') or 5
dev_questions_path = os.getenv('DEV_QUESTIONS_PATH') or '../data/dev_questions.csv'
chunks_800_path = os.getenv('CHUNKS_PATH') or '../data/output/chunks_800'
chunks_1000_path = os.getenv('CHUNKS_PATH') or '../data/output/chunks_1000'
pdf_info_path = os.getenv('PDF_INFO_PATH') or '../data/output/pdf_info'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,  # ← this is the key change
)

hybrid_800_retriever_path = '../data/uk_hybrid_index_800'
embedding_800_retriever_path = '../data/uk_embedding_index_800'
bm25_800_retriever_path = '../data/uk_bm25_index_800'

hybrid_1000_retriever_path = '../data/uk_hybrid_index_1000'
embedding_1000_retriever_path = '../data/uk_embedding_index_1000'
bm25_1000_retriever_path = '../data/uk_bm25_index_1000'

hybrid_800_e5large_retriever_path = '../data/uk_hybrid_index_800_e5large'
embedding_800_e5large_retriever_path = '../data/uk_embedding_index_800_e5large'

hybrid_1000_e5large_retriever_path = '../data/uk_hybrid_index_1000_e5large'
embedding_1000_e5large_retriever_path = '../data/uk_embedding_index_1000_e5large'

hybrid_800_bgem3_retriever_path = '../data/uk_hybrid_index_800_bgem3'
embedding_800_bgem3_retriever_path = '../data/uk_embedding_index_800_bgem3'

hybrid_1000_bgem3_retriever_path = '../data/uk_hybrid_index_1000_bgem3'
embedding_1000_bgem3_retriever_path = '../data/uk_embedding_index_1000_bgem3'

embedding_model_bge = 'BAAI/bge-m3'
embedding_model_base = 'intfloat/multilingual-e5-base'
embedding_model_large = 'intfloat/multilingual-e5-large'

model_name = os.getenv('MODEL_NAME') or 'lapa-llm/lapa-v0.1.2-instruct'

captured_logits_tmp_path = '/tmp/captured_logits.json'

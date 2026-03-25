import os


USE_VLLM = True if int(os.getenv('USE_VLLM', 0)) == 1 else False
VLLM_SERVE = True if int(os.getenv('VLLM_SERVE', 0)) == 1 else False
USE_RERANKER = True if int(os.getenv('USE_RERANKER', 0)) == 1 else False
dev_questions_path = os.getenv('DEV_QUESTIONS_PATH') or '../data/dev_questions.csv'
chunks_path = os.getenv('CHUNKS_PATH') or '../data/output/chunks_1000'
pdf_info_path = os.getenv('PDF_INFO_PATH') or '../data/output/pdf_info'
retriever_path = os.getenv('RETRIEVER_PATH') or '../data/uk_hybrid_index_1000_e5large'
embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME') or 'intfloat/multilingual-e5-large'
reranker_model_name = os.getenv('RERANKER_MODEL_NAME') or 'BAAI/bge-reranker-v2-m3'
llm_model_name = os.getenv('LLM_MODEL_NAME') or 'lapa-llm/lapa-v0.1.2-instruct'
retriever_device = os.getenv('RETRIEVER_DEVICE') or 'cpu'
llm_gpu_memory_utilization = os.getenv('LLM_GPU_MEMORY_UTILIZATION') or '0.5'

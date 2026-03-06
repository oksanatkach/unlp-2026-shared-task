from transformers import BitsAndBytesConfig
import torch
import os


YES_NO_QUESTIONS = os.getenv('YES_NO_QUESTIONS') or False
RETRIEVER_TOP_K = os.getenv('RETRIEVER_TOP_K') or 5
dev_questions_path = os.getenv('DEV_QUESTIONS_PATH') or 'data/dev_questions.csv'
chunks_path = os.getenv('CHUNKS_PATH') or 'data/output/chunks'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", # A special 4-bit data type
    bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 for faster computation
)
model_name = os.getenv('MODEL_NAME') or 'lapa-llm/lapa-v0.1.2-instruct'

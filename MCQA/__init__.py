from MCQA.device import load_method
from MCQA.objects import init

if load_method == 'vllm_server':
    from MCQA.question_answering_vllm import (
        answer_question_prompt_per_chunk_per_option_API as answer_question,
    )
elif load_method == 'VLLM':
    from MCQA.question_answering_vllm import (
        answer_question_prompt_per_chunk_per_option as answer_question,
    )
else:
    from MCQA.question_answering_transformers import (
        answer_question_prompt_per_chunk_per_option as answer_question,
    )

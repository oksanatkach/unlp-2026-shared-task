import logging
import os

from conf import config
from MCQA.utils import get_accelerator
device = get_accelerator()

logger = logging.getLogger(__name__)

load_method = None
if config.USE_VLLM and 'A100' in device:
    logger.info(f'Device is {device}, loading VLLM model')
    load_method = 'VLLM'

elif 'TPU' in device:
    if not config.USE_VLLM:
        logger.error(f"Device is {device}, can't use transformers, loading VLLM model")
    else:
        logger.info(f'Device is {device}, loading VLLM model')

    os.environ["PJRT_DEVICE"] = "TPU"
    load_method = 'VLLM'

elif config.USE_VLLM:
    logger.error(f"Can't use VLLM for device {device}, loading transformers model")
    load_method = 'transformers'
else:
    logger.info(f'Device is {device}, loading transformers model')
    load_method = 'transformers'

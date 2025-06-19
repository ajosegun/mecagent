import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from src.scripts.baseline import get_model_tokenizer, device
from src.utils.logging import get_logger
from src.config import Config

logger = get_logger()
config = Config()

checkpoint_dir = config.BASELINE_OUT_DIR  


# Load processor and model
processor = BlipProcessor.from_pretrained(checkpoint_dir)
model = BlipForConditionalGeneration.from_pretrained(checkpoint_dir)
model.to(device)
model.eval()

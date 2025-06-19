
import os
import torch
from src.utils.gencad_dataset import (
    preprocess_ds, 
    load_preprocessed_dataset,
    collate_fn
)
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.utils.logging import get_logger
from src.config import Config

logger = get_logger()
config = Config()

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def get_model_tokenizer():

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    model.to(device)  

    logger.info("Configure tokenizer and model config")
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.cls_token_id or tokenizer.bos_token_id or tokenizer.eos_token_id
    return model, processor, tokenizer
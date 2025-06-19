import pytest
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

def test_model_forward():
    encoder_model = "google/vit-base-patch16-224-in21k"
    decoder_model = "gpt2"
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model, decoder_model)
    feature_extractor = ViTImageProcessor.from_pretrained(encoder_model)
    tokenizer = AutoTokenizer.from_pretrained(decoder_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Dummy image and code
    import numpy as np
    from PIL import Image
    image = Image.fromarray(np.uint8(np.random.rand(224,224,3)*255))
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    labels = tokenizer("print('hello')", return_tensors="pt").input_ids

    outputs = model(pixel_values=pixel_values, labels=labels)
    assert outputs.loss is not None
    assert outputs.logits.shape[0] == 1
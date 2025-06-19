import pytest
import os
from transformers import BlipProcessor, AutoTokenizer
from src.utils import gencad_dataset

@pytest.fixture(scope="module")
def processor_and_tokenizer():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    tokenizer = processor.tokenizer 
    return processor, tokenizer

def test_load_hf_dataset():
    dataset = gencad_dataset.load_hf_dataset()
    assert "train" in dataset
    assert "test" in dataset
    assert "validation" in dataset
    assert len(dataset["train"]) > 0
    assert "image" in dataset["train"].features
    assert "cadquery" in dataset["train"].features

def test_preprocess_single_example(processor_and_tokenizer):
    processor, tokenizer = processor_and_tokenizer
    dataset = gencad_dataset.load_hf_dataset()
    example = dataset["train"][0]
    processed = gencad_dataset.preprocess(example, processor, tokenizer, max_target_length=128)
    assert "pixel_values" in processed
    assert "labels" in processed
    assert processed["pixel_values"].shape[0] in [3, 224, 224, 256, 512] 
    assert processed["labels"].ndim == 1

def test_preprocess_ds_and_load(tmp_path, processor_and_tokenizer):
    processor, tokenizer = processor_and_tokenizer
    dataset_path = tmp_path / "preprocessed_gencad"
    preprocessed = gencad_dataset.preprocess_ds(tokenizer, processor, str(dataset_path))
    assert "train" in preprocessed
    assert "test" in preprocessed
    assert os.path.exists(dataset_path)
    loaded = gencad_dataset.load_preprocessed_dataset(str(dataset_path))
    assert "train" in loaded
    assert len(loaded["train"]) == len(preprocessed["train"])
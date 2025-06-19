import os, json
from src.utils.logging import get_logger
from src.config import Config
from functools import partial
from PIL import Image
import torch

from datasets import load_dataset, DatasetDict, load_from_disk

config = Config()
logger = get_logger()

HF_DATASETS_CACHE = config.HF_DATASETS_CACHE

# Set custom cache directory
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE


def load_hf_dataset():
    logger.info("About to load dataset")
    dataset: DatasetDict = load_dataset(
        "CADCODER/GenCAD-Code",
        split={
            "train": "train[:500]",
            "test": "test[:100]",
            "validation": "validation[:100]"
        },
    cache_dir=HF_DATASETS_CACHE
)
    logger.info("Dataset loaded successfully")

    # Print summary
    for split_name, split_data in dataset.items():
        logger.info(f"\n--- {split_name.upper()} SPLIT ---")
        logger.info(f"Number of samples: {len(split_data)}")
        logger.info(f"Sample keys: {split_data[0].keys()}")
        sample = {k: str(split_data[0][k])[:100] for k in split_data[0]}
        logger.info(f"Sample item (truncated): {json.dumps(sample, indent=4)}")


    return dataset


def preprocess(example, processor, tokenizer, max_target_length):
    inputs = processor(
        images=example["image"],
        text=example["cadquery"],
        padding="max_length",
        truncation=True,
        max_length=max_target_length,
        return_tensors="pt"
    )
    pixel_values = inputs["pixel_values"][0]
    input_ids = inputs["input_ids"][0]    
    attention_mask = inputs["attention_mask"][0]  # tensor [seq_len]
    # mask pad tokens
    labels = torch.where(input_ids == processor.tokenizer.pad_token_id, -100, input_ids)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

    # return {"pixel_values": pixel_values, "labels": labels}



def preprocess_ds(tokenizer, processor, dataset_path):
    dataset = load_hf_dataset()

    logger.info("Preprocessing dataset...")
    preprocess_with_args = partial(
        preprocess,
        processor=processor,
        tokenizer=tokenizer,
        max_target_length=512
    )
    
    train_dataset = dataset["train"].map(preprocess_with_args, remove_columns=dataset["train"].column_names)
    test_dataset = dataset["test"].map(preprocess_with_args, remove_columns=dataset["test"].column_names)
    validation_dataset = dataset["validation"].map(preprocess_with_args, remove_columns=dataset["validation"].column_names)

    print(f"Train dataset size: {train_dataset[0].keys()}")
    _columns=["pixel_values", "input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=_columns)
    test_dataset.set_format(type="torch", columns=_columns)
    validation_dataset.set_format(type="torch", columns=_columns)

    # After preprocessing
    preprocessed = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
        "validation_dataset": validation_dataset
    })

    preprocessed.save_to_disk(dataset_path)

    logger.info(f"Preprocessing complete. Dataset saved to {dataset_path}.")
    return preprocessed


def load_preprocessed_dataset(dataset_path):
    try:
        dataset = load_from_disk(dataset_path)
        logger.info(f"{dataset_path} dataset loaded successfully - type {type(dataset)}.")
        return dataset
    except FileNotFoundError:
        logger.error(f"{dataset_path} dataset not found. Please run the preprocessing step first.")
        return None


def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])       
    input_ids    = torch.stack([x["input_ids"] for x in batch])          
    attention_mask = torch.stack([x["attention_mask"] for x in batch])   
    labels       = torch.stack([x["labels"] for x in batch])             
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
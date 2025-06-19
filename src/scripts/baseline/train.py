import os
import torch
from src.utils.gencad_dataset import (
    preprocess_ds, 
    load_preprocessed_dataset,
    collate_fn
)
from src.scripts.baseline import get_model_tokenizer, device
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.utils.logging import get_logger
from src.config import Config

logger = get_logger()
config = Config()

logger.info(f"Using device: {device}")

BASELINE_DATASET_PATH = config.BASELINE_DATASET_PATH

logger.info("Loading model and tokenizer...")
model, processor, tokenizer = get_model_tokenizer()

logger.info("Loading dataset ")
dataset = load_preprocessed_dataset(BASELINE_DATASET_PATH)

if not dataset:
    logger.info("Dataset not found. Preprocessing...")
    dataset = preprocess_ds(tokenizer, processor, BASELINE_DATASET_PATH)

train_dataset = dataset["train"]
test_dataset = dataset["test"]
eval_dataset = dataset.get("validation_dataset")


## BASELINE Model
logger.info("Configuring training arguments ")
# training_args = Seq2SeqTrainingArguments(
#     output_dir=config.BASELINE_OUT_DIR,
#     per_device_train_batch_size=1,  
#     per_device_eval_batch_size=1,
#     gradient_accumulation_steps=4,  
#     num_train_epochs=1,         
#     logging_steps=10,
#     eval_strategy="epoch", 
#     save_strategy="epoch",
#     predict_with_generate=False, 
#     fp16=False,
#     report_to="none",
#     save_total_limit=1,
# )

## IMPROVED
training_args = Seq2SeqTrainingArguments(
    output_dir=config.IMPROVED_OUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=2,                # try 3 epochs (instead of 1)
    learning_rate=3e-5,                # try a slightly lower LR
    weight_decay=0.01,                 # add small weight decay
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,        # enable generate() during eval
    generation_num_beams=4,             # use beam search
    load_best_model_at_end=True,       # keep best checkpoint by metric
    greater_is_better=True,
    fp16=False,                        # no mixed precision on MPS
    report_to="none",
    save_total_limit=1,
)

# Enable gradient checkpointing if memory is tight
model.gradient_checkpointing_enable()

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer,
)

logger.info("Pre-flight checks")
for batch in trainer.get_train_dataloader():
    print("Batch keys:", batch.keys())
    print("pixel_values type/shape:", type(batch["pixel_values"]), batch["pixel_values"].shape)
    print("labels type/shape:", type(batch["labels"]), batch["labels"].shape)
    break

logger.info("Starting training... This will take a while.")
trainer.train()
logger.info("Training completed")

logger.info("Evaluating ... ")
metrics = trainer.evaluate()
logger.info(f"Baseline metrics: {metrics}")


logger.info("Saving metrics ... ")
# with open(f"{config.BASELINE_METRICS_DIR}/baseline_metrics.txt", "w") as f:
#     f.write(str(metrics))

with open(f"{config.IMPROVED_METRICS_DIR}/improved_metrics.txt", "w") as f:
    f.write(str(metrics))

logger.info("All Done")
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from src.utils.gencad_dataset import (
    collate_fn, 
    load_preprocessed_dataset, 
    preprocess_ds,
    load_hf_dataset
)
from src.scripts.baseline import get_model_tokenizer, device
from src.utils.logging import get_logger
from src.config import Config

logger = get_logger()
config = Config()


# 3. Load processor and model
model, processor, tokenizer = get_model_tokenizer()


BASELINE_DATASET_PATH = config.BASELINE_DATASET_PATH
# Load dataset
dataset = load_preprocessed_dataset(BASELINE_DATASET_PATH)

if not dataset:
    logger.info("Dataset not found. Preprocessing...")
    dataset = preprocess_ds(tokenizer, processor, BASELINE_DATASET_PATH)

train_dataset = dataset["train"]
test_dataset = dataset["test"]
eval_dataset = dataset.get("validation_dataset")


if eval_dataset is None:
    raise ValueError("Validation dataset not found. Ensure you have a 'validation' split in your dataset.")

training_args = Seq2SeqTrainingArguments(
    # output_dir=config.BASELINE_OUT_DIR,
    output_dir=config.IMPROVED_OUT_DIR,
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    eval_strategy="epoch", 
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
    data_collator=collate_fn, 
)

# metrics = trainer.evaluate()
# logger.info(f"Baseline metrics: {metrics}")

# # Save metrics
# with open(f"{config.BASELINE_METRICS_DIR}/baseline_metrics.txt", "w") as f:
#     f.write(str(metrics))

from src.metrics.valid_syntax_rate import evaluate_syntax_rate_simple
from src.metrics.best_iou import get_iou_best

_ds = load_hf_dataset()
eval_ds = _ds["validation"].select(range(5))

ds_predict = eval_dataset.select(range(5))

pred_results = trainer.predict(ds_predict)
preds = pred_results.predictions 
decoded_preds = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)

codes = {}
for i, text in enumerate(decoded_preds):
    codes[f"Sample {i}"] = text
    print(f"\n\n Example {i}: {text}")

print("\n\n\n")
vsr = evaluate_syntax_rate_simple(codes)
print("Valid Syntax Rate:", vsr)


for expected, predicted in zip(eval_ds, list(codes.values())):
    try:
        iou = get_iou_best(expected["cadquery"], predicted)
        print("IOU:", iou)
    except Exception as e:
        print(e)
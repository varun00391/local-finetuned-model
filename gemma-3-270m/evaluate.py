from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import evaluate

# Load your test set
test_dataset = load_dataset("json", data_files="data/test.json")["train"]

# Load model & tokenizer
model = AutoModelForCausalLM.from_pretrained("./finetuned_model")
tokenizer = AutoTokenizer.from_pretrained("./finetuned_model")

# Example metric (for classification use 'accuracy')
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Reload Trainer with eval dataset
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=4,
    ),
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

results = trainer.evaluate()
print(results)

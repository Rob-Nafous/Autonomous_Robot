# Tuto from https://huggingface.co/docs/transformers/en/training
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM
import numpy as np
import evaluate
import torch

# Load dataset 
dataset = load_dataset("json", data_files="datasetSpotty.json")

# Tokenizer 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
# Define the padding token (necessary for training)
tokenizer.pad_token = tokenizer.eos_token

# Define the function for dataset tokenization
def tokenize_function(examples):
    texts = [
    f"<|user|>\n{q}\n<|assistant|>\n{r} {tokenizer.eos_token}"
    for q, r in zip(examples["question"], examples["réponse"])
    ]
    
    # Tokenization 
    tokenized = tokenizer(
        texts,
        padding="max_length",
        max_length=30,  
        truncation=True,
        return_tensors="pt"
    )

    # Create labels by shifting the input IDs to the right
    labels = tokenized["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100  # We ignore padding tokens in the loss

    tokenized["labels"] = labels

    return tokenized

# Apply it to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]  # Only train dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype="auto")

# Training hyperparameters
training_args = TrainingArguments(
    output_dir="test_trainer",
    num_train_epochs=8,  # Commence avec 5 époques, ajuste après test
    per_device_train_batch_size=7,  # Augmente si tu as plus de VRAM
    evaluation_strategy="no",  # Limit the total amount of checkpoints
    report_to="none",  # We do not want to report to the Hugging Face Hub
)

# Evaluate 
metric = evaluate.load("accuracy")
print(metric)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,  # Ensure tokenizer is passed to the Trainer
    compute_metrics=compute_metrics  # Include evaluation metrics
)

trainer.train()

# Save the model and tokenizer
trainer.save_model("test_trainer")  # Save the model correctly
tokenizer.save_pretrained("test_trainer")  # Save the tokenizer as well
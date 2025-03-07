#Tuto from https://huggingface.co/docs/transformers/en/training
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM
import numpy as np
import evaluate
import torch



#Load dataset 
dataset = load_dataset("json", data_files="dataset.json")

#Tokenizer 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
# Définir le token de padding (nécessaire pour l'entraînement)
tokenizer.pad_token = tokenizer.eos_token

# Define the function for dataset tokenization
def tokenize_function(examples):
    texts = [
        f"<|system|>\nTu es un assistant utile.\n<|user|>\n{q}\n<|assistant|>\n{r}"
        for q, r in zip(examples["question"], examples["réponse"])
    ]
    
    # Tokenization avec padding dynamique
    tokenized = tokenizer(
        texts,
        padding= True,  
        truncation=True,
        return_tensors="pt"
    )

    tokenized["labels"] = tokenized["input_ids"].clone()  # Clone des labels pour l'apprentissage

    return tokenized


# Apply it to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]  # Uniquement train dataset

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype="auto")
#Training hyperparameters
training_args = TrainingArguments(
    output_dir="test_trainer",
    save_total_limit=2,
    num_train_epochs=3,
    report_to="none",
)

#Evaluate 
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# Sauvegarde manuelle du modèle
trainer.save_model("test_trainer")  # Enregistre le modèle correctement
tokenizer.save_pretrained("test_trainer")  # Enregistre aussi le tokenizer

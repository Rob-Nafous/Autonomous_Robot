from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Charger le modèle fine-tuné
model_path = "test_trainer"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Générer une réponse
question = "As-tu déjà voyagé à l’étranger ?"
input_text = f"<|user|>\n{question}\n<|assistant|>\n"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

output = model.generate(input_ids, max_new_tokens=50)
response = tokenizer.decode(output[0], skip_special_tokens=True)

print(response)

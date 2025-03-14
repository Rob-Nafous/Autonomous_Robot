from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Ou load_in_8bit=True si tu veux un meilleur compromis vitesse/qualité
    bnb_4bit_compute_dtype=torch.float16
)

# Charger le modèle fine-tuné
model_path = "test_trainer"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto",quantization_config=quantization_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = torch.compile(model)

# Générer une réponse
question = "Par quel nom je dois t'appeler ?"
input_text = f"<|user|>\n{question}\n<|assistant|>\n"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

output = model.generate(input_ids, max_new_tokens=30,use_cache=True)
response = tokenizer.decode(output[0], skip_special_tokens=True)

print(response)


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Ou load_in_8bit=True  (need to see the difference)
    bnb_4bit_compute_dtype=torch.float16
)

# CFinetuned model
model_path = "Spotty"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto",quantization_config=quantization_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# HIstory exhanges (gotta change it for a .db format)
conversation_history = ""

while True:
    # Demander une entrée utilisateur
    user_input = input("👤 Vous : ")

    # Quitter si l'utilisateur tape "exit"
    if user_input.lower() == "exit":
        print("👋 Fin de la conversation. À bientôt !")
        break

    # Construire le format conversationnel
    conversation_history += f"<|user|>\n{user_input}\n<|assistant|>\n"

    # Tokenization et génération de réponse
    input_ids = tokenizer(conversation_history, return_tensors="pt").input_ids.to("cuda")
    
    output = model.generate(input_ids, max_new_tokens=50, use_cache=True)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extraire uniquement la nouvelle réponse (sans historique répété)
    assistant_response = response.split("<|assistant|>\n")[-1].strip()

    # Ajouter la réponse à l'historique
    conversation_history += f"{assistant_response}\n"

    # Afficher la réponse de l'IA
    print(f"🤖 Assistant : {assistant_response}\n")


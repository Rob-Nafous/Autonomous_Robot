import gradio as gr
import cv2
import numpy as np
import face_recognition
import os
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from pymongo.server_api import ServerApi

# Connexion MongoDB
uri = os.getenv('MONGODBPASSCODE')
client = MongoClient(uri, server_api=ServerApi('1'))
collection = client['users']['faces']

# Assurer l'unicité des usernames
if "username_1" not in collection.index_information():
    collection.create_index("username", unique=True)

# Chargement initial des embeddings
user_embeddings = {}

def load_embeddings():
    global user_embeddings
    user_embeddings.clear()  # Reset pour éviter doublons
    for user in collection.find():
        username = user['username']
        embedding = np.array(user['embedding'], dtype=np.float64)  # Convertir en NumPy array
        user_embeddings[username] = {'face': embedding}
    print(f"Loaded {len(user_embeddings)} users.")

load_embeddings()

def recognize_user(image):
    img = np.array(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    embeddings_unknown = face_recognition.face_encodings(img_rgb)
    if not embeddings_unknown:
        return "No face detected.", None, None

    recognized_names = []
    unknown_embeddings = []

    for unknown_embedding in embeddings_unknown:
        name, similarity_score = recognize(unknown_embedding)
        if name not in ['unknown_person', 'no_persons_found']:
            confidence = (1 - similarity_score) * 100  # Convertir en %
            recognized_names.append(f"{name} ({confidence:.2f}%)")
        else:
            unknown_embeddings.append(unknown_embedding)

    if not recognized_names and not unknown_embeddings:
        return "Unknown user. Please enter your name to register.", None, None

    if unknown_embeddings:
        return "Hello " + ", ".join(recognized_names) + ". Please enter the name for the unknown person.", unknown_embeddings[0], recognized_names

    return "Hello " + ", ".join(recognized_names), None, recognized_names

def register_user(name, embedding):
    if not name:
        return "Error: Name field is empty."
    if name in user_embeddings:
        return "Error: Name already taken."

    try:
        embedding_list = embedding.tolist()
        document = {'username': name, 'embedding': embedding_list}
        collection.insert_one(document)

        # Mettre à jour user_embeddings en direct
        user_embeddings[name] = {'face': np.array(embedding_list, dtype=np.float64)}

        return f"User {name} registered successfully!"
    except DuplicateKeyError:
        return f"Error: The username '{name}' is already registered."
    except Exception as e:
        return f"Error registering user: {str(e)}"

def recognize(unknown_embedding):
    threshold = 0.6
    min_dis_id = 'unknown_person'
    min_distance = threshold

    for username, embedding in user_embeddings.items():
        face_embedding = np.array(embedding['face'], dtype=np.float64)
        distance = np.linalg.norm(face_embedding - unknown_embedding)
        if distance < min_distance:
            min_distance = distance
            min_dis_id = username

    return min_dis_id, min_distance

def process_frame(image, name):
    message, embedding, recognized_names = recognize_user(image)
    if embedding is not None and name:
        registration_message = register_user(name, embedding)
        return f"{message} {registration_message}"
    return message

# Interface Gradio
iface = gr.Interface(
    fn=process_frame,
    inputs=[
        gr.Image(source="webcam", streaming=True, label="Camera"), 
        gr.Textbox(label="Enter name for unknown person")
    ],
    outputs="text",
    live=True,
    title="Real-Time Face Recognition and Registration",
    allow_flagging='never'
)

if __name__ == "__main__":
    iface.launch()

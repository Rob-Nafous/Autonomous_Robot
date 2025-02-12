from pveagle import EagleProfile
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import cv2
import numpy as np
import face_recognition
from PIL import Image
import gradio as gr
import os

uri = os.getenv('MONGODBPASSCODE')
client = MongoClient(uri, server_api=ServerApi('1'))
collection = client['users']['faces']
collection.create_index("username", unique=True)

# Load all embeddings into memory
user_embeddings = {}

def load_embeddings():
    global user_embeddings
    user_embeddings = {}
    for user in collection.find():
        stats = {}
        username = user['username']
        embedding = np.array(user['embedding'], dtype=np.float64)  # Ensure NumPy array
        stats['face'] = embedding
        user_embeddings[username] = stats
        print(username)

load_embeddings()

def recognize_user(image):
    img = np.array(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    embeddings_unknown = face_recognition.face_encodings(img_rgb)
    if len(embeddings_unknown) == 0:
        return "No person detected"
    
    name, similarity_score = recognize(img_rgb)
    if name in ['unknown_person', 'no_persons_found']:
        return "Unknown user. Please register if you have not or try again."
    
    confidence = (1 - similarity_score) * 100  # Score en %
    return f"Hello {name}, I'm {confidence:.2f}% sure it's you."

def register_user(image, name):
    if not name:
        return "Name field empty."
    elif name in user_embeddings:
        return "Name already taken."

    img = np.array(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    embeddings = face_recognition.face_encodings(img_rgb)

    if len(embeddings) == 0:
        return "No face detected in the image."

    embedding = np.array(embeddings[0], dtype=np.float64)  # Ensure NumPy array

    document = {
        'username': name,
        'embedding': embedding.tolist()
    }
    collection.insert_one(document)
    user_embeddings[name] = {}
    user_embeddings[name]['face'] = embedding

    return f"User {name} registered successfully!"

def delete_user(name):
    if not name:
        return "Name field empty."

    result = collection.delete_one({'username': name})
    if result.deleted_count > 0:
        user_embeddings.pop(name)
        return f'User {name} deleted successfully!'
    else:
        return f'User {name} not found!'

def recognize(img):  
    embeddings_unknown = face_recognition.face_encodings(img)
    if len(embeddings_unknown) == 0:
        return 'no_persons_found', 1.0

    unknown_embedding = np.array(embeddings_unknown[0], dtype=np.float64)

    threshold = 0.6
    min_dis_id = 'unknown_person'
    min_distance = threshold

    for username, embedding in user_embeddings.items():
        face_embedding = np.array(embedding['face'], dtype=np.float64)
        distance = np.linalg.norm(face_embedding - unknown_embedding)  # Corriger ici
        if distance < min_distance:
            min_distance = distance
            min_dis_id = username
            
    return min_dis_id, min_distance

theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont('Roboto'), gr.themes.GoogleFont('Montserrat'), 'ui-sans-serif', 'sans-serif']
)

iface_recognize = gr.Interface(
    fn=recognize_user,
    inputs=[gr.Image(label="Show your whole face to the camera", source="webcam", streaming=True)],
    outputs=[gr.HTML()],
    live=True,
    title="Face Recognition Attendance System",
    allow_flagging='never'
)

iface_register = gr.Interface(
    fn=register_user,
    inputs=[gr.Image(label="Ensure your face is properly shown and details are entered below", source="webcam", streaming=True),
            gr.Textbox(label="Enter new user name")],
    outputs=[gr.HTML()],
    title="Register New User",
    live=False,
    allow_flagging='never'
)

iface_delete = gr.Interface(
    fn=delete_user,
    inputs=[gr.Textbox(label="Enter user name to delete")],
    outputs=[gr.HTML()],
    title="Delete User",
    live=False,
    allow_flagging='never'
)

custom_css = """
    footer {display: none !important;}
    div.stretch button.secondary {display: none !important;}
    .panel .pending {opacity: 1 !important;}
    .tab-nav button {font-size: 1.5rem !important;}
    .prose {font-size: 3rem !important;}
    .gap.panel {border: none !important;}
"""

iface = gr.TabbedInterface([iface_register, iface_recognize, iface_delete], ["Register User", "Verify User", "Delete User"],
                           css=custom_css, theme=theme)

if __name__ == "__main__":
    iface_recognize.dependencies[0]["show_progress"] = False
    iface_register.dependencies[0]["show_progress"] = False
    iface_delete.dependencies[0]["show_progress"] = False
    iface.launch()

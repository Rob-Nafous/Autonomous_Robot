import numpy as np
import sqlite3
from insightface.app import FaceAnalysis
from picamera2 import Picamera2
import cv2
import os
from pynput import keyboard
import queue

# --------- Camera ---------
picam2 = Picamera2()
config = picam2.create_preview_configuration()
picam2.configure(config)
picam2.start()

# --------- SQLite Database ---------
db_name = 'faces.db'
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Create users table for storing known faces
cursor.execute('''CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    embedding BLOB NOT NULL)''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages ( 
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sender TEXT NOT NULL,
        recipient TEXT,
        message TEXT,
        FOREIGN KEY(sender) REFERENCES users(name),
        FOREIGN KEY(recipient) REFERENCES users(name)
    )
''')
conn.commit()

# --------- InsightFace Initialization ---------
print("🔄 Chargement du modèle InsightFace...")
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)

# --------- SQL Functions ---------
def add_face(face_embedding, name):
    #Name already exists ? 
    cursor.execute('SELECT embedding FROM faces WHERE name=?',(name,))
    rows = cursor.fetchall()
    
    if rows:
        embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in rows]
        embeddings.append(face_embedding)
        print("My bad ! I'll do my best to recognize you more")
        if len(embeddings) > 5:
            embeddings.pop(0)
        cursor.execute('DELETE FROM faces WHERE name=?',(name,))
        for emb in embeddings:
            cursor.execute('INSERT INTO faces (name, embedding) VALUES (?, ?)', (name, emb.tobytes()))
            
    else:
        cursor.execute('INSERT INTO faces (name, embedding) VALUES (?, ?)', (name, face_embedding.tobytes()))
        print(f"{name}, I'll try to remember you! ")
    #Then add it
    conn.commit()

def recognize_face(face_embedding):
    cursor.execute('SELECT id, name, embedding FROM faces')
    rows = cursor.fetchall()
    
    for row in rows:
        stored_name = row[1]
        stored_embedding = np.frombuffer(row[2], dtype=np.float32)
        
        similarity = np.dot(face_embedding, stored_embedding) / (np.linalg.norm(face_embedding) * np.linalg.norm(stored_embedding)) 
        if similarity > 0.7: 
            return stored_name
    return None

# --------- Messages Functions ---------
def add_message(sender, recipient, message):
    cursor.execute("INSERT INTO messages (sender, recipient, message) VALUES (?, ?, ?)", (sender, recipient, message))
    conn.commit()
    print(f"Message saved for {recipient}: {message}")

message_queue = queue.Queue()

def on_press(key):
    try:
        if key.char == 'm':
            recipient = input("To whom do you want to send a message? ")
            message = input("What's the message? ")
            sender = input("From whom? ")
            message_queue.put((sender, recipient, message))
    except AttributeError:
        pass

def read_messages(recipient):
    cursor.execute("SELECT sender, message FROM messages WHERE recipient=?", (recipient,))
    messages = cursor.fetchall()

    if messages:
        for sender, msg in messages:
            print(f"Message from {sender}: {msg}")

        # Delete messages after reading
        cursor.execute("DELETE FROM messages WHERE recipient=?", (recipient,))
        conn.commit()

# Start the keyboard listener in non-blocking mode
listener = keyboard.Listener(on_press=on_press)
listener.start()
# --------- Main ---------

print("Press 'm' to write a message")

def detect_and_recognize():
    while True:

        while not message_queue.empty():
            sender, recipient, message = message_queue.get()
            add_message(sender, recipient, message)

        im = picam2.capture_array()
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        faces = app.get(im)

        for face in faces:
            face_embedding = face.embedding
            name = recognize_face(face_embedding)
            
            if name:
                print(f"I see {name}'s face!")
                read_messages(name)
            else:
                print("Nice to meet you!")
                name = input("What's your name? ")
                add_face(face_embedding, name)
                

detect_and_recognize()

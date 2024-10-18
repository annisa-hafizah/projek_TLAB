import sqlite3
from flask import Flask, request, jsonify
from deepface import DeepFace
import numpy as np
import faiss
import base64
import cv2

app = Flask(__name__)

# Path to SQLite database
db_path = './face_embeddings.db'

# Initialize SQLite database and create table if it doesn't exist
def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            position TEXT,
            embedding BLOB
        )
    ''')
    conn.commit()
    conn.close()

# Insert employee data into SQLite
def insert_employee(name, position, embedding):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO employees (name, position, embedding) VALUES (?, ?, ?)", 
                   (name, position, embedding))
    conn.commit()
    conn.close()

# Retrieve all embeddings from the SQLite database
def fetch_embeddings():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, position, embedding FROM employees")
    data = cursor.fetchall()
    conn.close()
    return data

# Convert embeddings stored as blob back to numpy array
def convert_embedding(blob):
    return np.frombuffer(blob, dtype='f')

@app.route('/register-employee', methods=['POST'])
def register_employee():
    data = request.json
    name = data['name']
    position = data['position']
    image_base64 = data['image']

    model_name = "Facenet"
    detector_backend = "opencv"

    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_base64)
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Generate embedding
        objs = DeepFace.represent(
            img_path=img,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=False
        )

        if len(objs) > 0:
            embedding = np.array(objs[0]['embedding'], dtype='f')
            embedding_blob = embedding.tobytes()  # Convert numpy array to binary
            insert_employee(name, position, embedding_blob)  # Save employee to DB
            return jsonify({"message": "Karyawan berhasil didaftarkan!"}), 200
        else:
            return jsonify({"message": "Tidak ada wajah terdeteksi."}), 400

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.route('/identify-employee', methods=['POST'])
def identify_employee():
    image_base64 = request.json['image']
    model_name = "Facenet"
    detector_backend = "opencv"

    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_base64)
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Generate embedding
        target_embedding = DeepFace.represent(
            img_path=img,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=False
        )[0]["embedding"]

        target_embedding = np.expand_dims(np.array(target_embedding, dtype='f'), axis=0)

        # Fetch all embeddings from the database
        employees_data = fetch_embeddings()

        # Extract embeddings and store them in a numpy array
        embeddings = []
        employee_details = []
        for emp in employees_data:
            name, position, embedding_blob = emp
            embedding = convert_embedding(embedding_blob)
            embeddings.append(embedding)
            employee_details.append((name, position))

        embeddings = np.array(embeddings, dtype='f')

        # Initialize FAISS index and search for the closest match
        num_dimensions = 128
        index = faiss.IndexFlatL2(num_dimensions)
        index.add(embeddings)
        k = 1  # Find the closest match
        distances, neighbours = index.search(target_embedding, k)

        if neighbours[0][0] < len(employee_details):
            match_name, position = employee_details[neighbours[0][0]]
            return jsonify({"name": match_name, "position": position}), 200

        return jsonify({"message": "Tidak ada kecocokan ditemukan."}), 404

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.route('/employees', methods=['GET'])
def get_employees():
    try:
        employees_data = fetch_embeddings()
        if employees_data:
            employees_list = [{'name': emp[0], 'position': emp[1]} for emp in employees_data]
            return jsonify(employees_list), 200
        else:
            return jsonify({"message": "Tidak ada data karyawan."}), 404
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500
    
@app.route('/update-employee/<int:id>', methods=['PUT'])
def update_employee(id):
    data = request.json
    name = data.get('name')
    position = data.get('position')

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if employee exists
        cursor.execute("SELECT * FROM employees WHERE id = ?", (id,))
        employee = cursor.fetchone()

        if employee:
            # Update employee details
            cursor.execute("""
                UPDATE employees
                SET name = ?, position = ?
                WHERE id = ?
            """, (name, position, id))
            conn.commit()
            return jsonify({"message": "Karyawan berhasil diperbarui!"}), 200
        else:
            return jsonify({"message": "Karyawan tidak ditemukan."}), 404

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

    finally:
        conn.close()

@app.route('/delete-employee/<int:id>', methods=['DELETE'])
def delete_employee(id):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if employee exists
        cursor.execute("SELECT * FROM employees WHERE id = ?", (id,))
        employee = cursor.fetchone()

        if employee:
            # Delete employee
            cursor.execute("DELETE FROM employees WHERE id = ?", (id,))
            conn.commit()
            return jsonify({"message": "Karyawan berhasil dihapus!"}), 200
        else:
            return jsonify({"message": "Karyawan tidak ditemukan."}), 404

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

    finally:
        conn.close()


if __name__ == '__main__':
    init_db()  # Initialize the database
    app.run(debug=True, host='127.0.0.1', port=5000)
from flask import Flask, request, jsonify
from deepface import DeepFace
import pandas as pd
import numpy as np
import faiss
import base64
import cv2
import os

app = Flask(__name__)

# Folder untuk menyimpan embedding
embedding_folder = "./"
embedding_file = os.path.join(embedding_folder, "face_embeddings.csv")

# Fungsi untuk membuat embedding dari data karyawan
def create_embeddings(data):
    df = pd.DataFrame(columns=["name", "embedding", "posisi"])
    model_name = "Facenet"
    detector_backend = "opencv"

    for entry in data:
        name = entry["name"]
        posisi = entry["position"]
        image_base64 = entry["image"]

        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_base64)
            img_array = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Generate embedding wajah
            objs = DeepFace.represent(
                img_path=img,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False
            )

            if len(objs) > 0:
                for obj in objs:
                    embedding = obj["embedding"]
                    new_row = pd.DataFrame({"name": [name], "embedding": [embedding], "posisi": [posisi]})
                    df = pd.concat([df, new_row], ignore_index=True)
            else:
                print(f"No embeddings found for {name}")

        except Exception as e:
            print(f"Error processing {name}: {e}")

    if not df.empty:
        if os.path.exists(embedding_file):
            existing_df = pd.read_csv(embedding_file)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_csv(embedding_file, index=False)
        print("Embeddings telah disimpan dengan nama file face_embeddings.csv")
    else:
        print("Tidak ada embedding yang disimpan. Pastikan file gambar memiliki wajah.")

def create_faiss_index():
    df = pd.read_csv(embedding_file)
    df['embedding'] = df['embedding'].apply(eval)
    embeddings = np.array(df['embedding'].tolist(), dtype='f')
    num_dimensions = 128
    index = faiss.IndexFlatL2(num_dimensions)
    index.add(embeddings)
    faiss.write_index(index, "./faiss_index.bin")
    print("FAISS index created and saved to faiss_index.bin")

@app.route('/register-employee', methods=['POST'])
def register_employee():
    data = request.json
    create_embeddings([data])
    create_faiss_index()
    return jsonify({"message": "Karyawan berhasil didaftarkan!"}), 200

@app.route('/employees', methods=['GET'])
def get_employees():
    if os.path.exists(embedding_file):
        df = pd.read_csv(embedding_file)
        employees_list = df[['name', 'posisi']].to_dict(orient='records')
        return jsonify(employees_list), 200
    else:
        return jsonify({"message": "Tidak ada data karyawan."}), 404

@app.route('/identify-employee', methods=['POST'])
def identify_employee():
    image_base64 = request.json['image']
    model_name = "Facenet"
    detector_backend = "opencv"
    df = pd.read_csv(embedding_file)
    # Decode base64 image
    image_bytes = base64.b64decode(image_base64)
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Generate embedding wajah
    target_embedding = DeepFace.represent(
        img_path=img,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=False
    )[0]["embedding"]

    # Convert embedding ke numpy dan expand dimensinya
    target_embedding = np.expand_dims(np.array(target_embedding, dtype='f'), axis=0)

    # Cari embedding terdekat di FAISS
    index = faiss.read_index("./faiss_index.bin")
    k = 1
    distances, neighbours = index.search(target_embedding, k)

    if neighbours[0][0] < len(df):
        match_name = df.iloc[neighbours[0][0]]['name']
        position = df.iloc[neighbours[0][0]]['posisi']
        return jsonify({"name": match_name, "position": position}), 200

    return jsonify({"message": "Tidak ada kecocokan ditemukan."}), 404

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)

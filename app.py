import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS  # Add CORS support
import sqlite3
from deepface import DeepFace
import numpy as np
import faiss
import base64
import cv2
import json
from datetime import datetime, timedelta
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import os
import logging

load_dotenv()

app = Flask(__name__)
CORS(app) if os.getenv('CORS_ENABLED', 'True').lower() == 'true' else None

# Konfigurasi logging custom
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Menonaktifkan log default Flask
logging.getLogger('werkzeug').setLevel(logging.ERROR)


# Path to SQLite database
db_path = os.getenv('DB_PATH')
model_name = os.getenv('FACE_RECOGNITION_MODEL')
detector_backend = os.getenv('FACE_DETECTOR_BACKEND')
recognition_threshold = float(os.getenv('FACE_RECOGNITION_THRESHOLD'))

# Initialize SQLite database and create tables if they don't exist
def init_db():
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Table for storing embeddings (already exists)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                position TEXT,
                embedding BLOB
            )
        ''')
        
        #new table for attendance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_name TEXT NOT NULL,
                date DATE NOT NULL,
                jam_masuk TIME,
                jam_keluar TIME DEFAULT NULL,
                jam_kerja TEXT DEFAULT NULL,
                image_capture TEXT,
                status TEXT NOT NULL
            )
        ''')
        conn.commit()

        # New table for storing employee images and info
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employee_info (
                employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                position TEXT,
                image_base64 TEXT
            )
        ''')
        conn.commit()

# Insert employee data into both employee_info and employees tables
def insert_employee(name, position, embedding, image_base64):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Insert into employees (embedding table)
        cursor.execute("INSERT INTO employees (name, position, embedding) VALUES (?, ?, ?)", 
                       (name, position, embedding))

        # Insert into employee_info (image table)
        cursor.execute("INSERT INTO employee_info (name, position, image_base64) VALUES (?, ?, ?)", 
                       (name, position, image_base64))
        
        conn.commit()

def fetch_embeddings():
       with sqlite3.connect(db_path) as conn:
           cursor = conn.cursor()
           cursor.execute("SELECT id, name, position, embedding FROM employees")
           data = cursor.fetchall()

           encoded_data = []
           for row in data:
               id, name, position, embedding_blob = row
               embedding_base64 = base64.b64encode(embedding_blob).decode('utf-8')
               encoded_data.append({
                   "id": id,
                   "name": name,
                   "position": position,
                   "embedding": embedding_base64
               })
       
       return encoded_data

# Convert embeddings stored as blob back to numpy array
def convert_embedding(blob):
    return np.frombuffer(blob, dtype='f')

class EnhancedFaceRecognition:
    def __init__(self, threshold=os.getenv('FACE_RECOGNITION_THRESHOLD')):
        self.threshold = threshold
        self.index = None
        self.employee_details = []
    
    def build_index(self, embeddings, details):
        """Build FAISS index with embeddings"""
        self.embeddings = np.array(embeddings, dtype='float32')
        self.employee_details = details
        
        # Initialize FAISS index
        num_dimensions = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(num_dimensions)
        self.index.add(self.embeddings)
        
        # Initialize sklearn NearestNeighbors for cosine similarity
        self.nn = NearestNeighbors(n_neighbors=1, metric='cosine')
        self.nn.fit(self.embeddings)
    
    def identify(self, target_embedding):
        """
        Identify a person using multiple similarity metrics and threshold
        Returns: (name, position, confidence) or ("Unknown", None, None)
        """
        target_embedding = np.array([target_embedding], dtype='float32')
        
        # Get L2 distance using FAISS
        distances, indices = self.index.search(target_embedding, 1)
        l2_distance = distances[0][0]
        
        # Get cosine similarity using sklearn
        distances_cosine, indices_cosine = self.nn.kneighbors(target_embedding)
        cosine_distance = distances_cosine[0][0]
        
        # Calculate normalized score (0-1, higher is better)
        l2_score = 1 / (1 + l2_distance)
        cosine_score = 1 - cosine_distance
        
        # Combined confidence score
        confidence = (l2_score + cosine_score) / 2
        
        # If confidence is below threshold, return Unknown
        if confidence < self.threshold:
            return "Unknown", None, confidence
            
        # Get matching employee details
        match_idx = indices[0][0]
        name, position = self.employee_details[match_idx]
        
        return name, position, confidence

@app.after_request
def custom_log(response):
    # Dapatkan informasi IP, metode, path, dan kode status
    ip_address = request.remote_addr
    method = request.method
    path = request.path
    status_code = response.status_code

    # Tentukan pesan sesuai dengan status kode
    if status_code == 200:
        message = "berhasil mendapatkan data employees."
    elif status_code == 404:
        message = "data tidak ditemukan."
    else:
        message = response.status

    # Format log
    log_message = f'{ip_address} - - [{datetime.now().strftime("%d/%b/%Y %H:%M:%S")}] "{method} {path} HTTP/1.1" {status_code} {message}'
    app.logger.info(log_message)

    return response

@app.route('/register-employee', methods=['POST'])
def register_employee():
    data = request.json
    name = data['name']
    position = data['position']
    image_base64 = data['image']  # The base64 image data

    model_name = "Facenet"
    detector_backend = "opencv"

    try:
        # Decode base64 image to process it with DeepFace
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

            # Insert employee into both tables (embedding and image data)
            insert_employee(name, position, embedding_blob, image_base64)

            return jsonify({"message": "Karyawan berhasil didaftarkan!"}), 200
        else:
            return jsonify({"message": "Tidak ada wajah terdeteksi."}), 400

    except Exception as e:
        print(f"Error during registration: {str(e)}")
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.route('/identify-employee', methods=['POST'])
def identify_employee():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"message": "No image data provided"}), 400

    try:
        image_base64 = data['image']
        image_bytes = base64.b64decode(image_base64)
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"message": "Failed to decode image"}), 400

        target_embedding = DeepFace.represent(
            img_path=img,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False
        )[0]["embedding"]

        # Fetch embeddings and create recognition system
        employees_data = fetch_embeddings()
        embeddings = []
        employee_details = []
        
        for emp in employees_data:
            embedding = np.frombuffer(base64.b64decode(emp['embedding']), dtype='f')
            embeddings.append(embedding)
            employee_details.append((emp['name'], emp['position']))

        recognition = EnhancedFaceRecognition(recognition_threshold)
        recognition.build_index(embeddings, employee_details)
        
        name, position, confidence = recognition.identify(target_embedding)
        
        if name == "Unknown":
            name = "Unknown Person"
            # Insert the unknown person's details into attendance logs
            status = 'masuk'  # or 'keluar' based on your system
            record_attendance({"name": name, "image": image_base64, "status": status})
            
        return jsonify({
            "name": name,
            "position": position,
            "confidence": float(confidence)
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

    
@app.route('/employees', methods=['GET'])
def get_employees():
    try:
        employees_data = fetch_embeddings()
        if employees_data:
            return jsonify(employees_data), 200
        else:
            return jsonify({"message": "Tidak ada data karyawan."}), 404
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500
    
@app.route('/employees-full', methods=['GET'])
def get_employees_full():
       try:
           with sqlite3.connect(db_path) as conn:
               cursor = conn.cursor()
               cursor.execute("SELECT id, name, position, embedding FROM employees")
               data = cursor.fetchall()

               employees_data = []
               for row in data:
                   id, name, position, embedding_blob = row
                   embedding_base64 = base64.b64encode(embedding_blob).decode('utf-8')
                   employees_data.append({
                       "id": id,
                       "name": name,
                       "position": position,
                       "embedding": embedding_base64
                   })

           return jsonify(employees_data), 200
       except Exception as e:
           return jsonify({"message": f"Error: {str(e)}"}), 500
    
# Fetch all employee info from the employee_info table
@app.route('/employees-info', methods=['GET'])
def get_employees_info():
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT employee_id, name, position, image_base64 FROM employee_info")
            data = cursor.fetchall()

            employee_list = []
            for row in data:
                employee_id, name, position, image_base64 = row
                employee_list.append({
                    'employee_id': employee_id,
                    'name': name,
                    'position': position,
                    'image_base64': image_base64
                })

            return jsonify(employee_list), 200
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

#menambah get time period
def get_time_period():
    """Return the time period based on current hour"""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "pagi"
    elif 12 <= hour < 15:
        return "siang"
    elif 15 <= hour < 18:
        return "sore"
    else:
        return "malam"

def is_late_arrival():
    """Check if current time is past the maximum arrival time (11:00)"""
    max_arrival_time = datetime.now().replace(hour=11, minute=0, second=0, microsecond=0)
    return datetime.now() > max_arrival_time

def calculate_working_hours(jam_masuk, jam_keluar=None):
    """Calculate working hours between check-in and current time"""
    jam_masuk_time = datetime.strptime(jam_masuk, '%H:%M:%S')

    if jam_keluar:
        jam_keluar_time = datetime.strptime(jam_keluar, '%H:%M:%S')
    else:
        jam_keluar_time = datetime.now().time()

    jam_keluar_time_full = datetime.strptime(jam_keluar_time.strftime('%H:%M:%S'), '%H:%M:%S')
    
    time_diff = jam_keluar_time_full - jam_masuk_time
    return time_diff

#menambah endpoint attendance
@app.route('/record-attendance', methods=['POST'])
def record_attendance():
    data = request.get_json()
    if not data:
        return jsonify({"message": "No data provided"}), 400

    employee_name = data['name']
    image_capture = data['image']
    status = data.get('status', 'masuk')
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            current_date = datetime.now().strftime('%Y-%m-%d')
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Check if it's late arrival (after 11:00)
            is_late = False
            if status == 'masuk':
                current_datetime = datetime.now()
                max_arrival = current_datetime.replace(hour=11, minute=0, second=0)
                is_late = current_datetime > max_arrival

            # If it's an unknown person, record only the "masuk" status
            if employee_name == "Unknown Person":
                cursor.execute("""
                    INSERT INTO attendance (
                        employee_name, date, jam_masuk, image_capture, status
                    )
                    VALUES (?, ?, ?, ?, ?)
                """, (employee_name, current_date, current_time, image_capture, status))
                
            else:
                # Check if employee already has an entry for today
                cursor.execute("""
                    SELECT id, jam_masuk, status, jam_keluar 
                    FROM attendance 
                    WHERE employee_name = ? AND date = ? 
                    ORDER BY jam_masuk DESC LIMIT 1
                """, (employee_name, current_date))
                
                existing_record = cursor.fetchone()
                
                if existing_record:
                    record_id, jam_masuk, current_status, jam_keluar = existing_record
                    
                    # Handle checkout - only if there's no previous checkout and status is 'masuk'
                    if status == 'keluar' and current_status == 'masuk' and jam_keluar is None:
                        # Calculate time difference
                        jam_masuk_time = datetime.strptime(jam_masuk, '%H:%M:%S')
                        current_time_obj = datetime.strptime(current_time, '%H:%M:%S')
                        
                        # Calculate elapsed time in minutes
                        elapsed_time = current_time_obj - jam_masuk_time
                        elapsed_minutes = elapsed_time.total_seconds() / 60
                        
                        # Only allow checkout if more than 10 minutes have passed
                        if elapsed_minutes >= 10:
                            jam_kerja = str(elapsed_time)
                            cursor.execute("""
                                UPDATE attendance 
                                SET jam_keluar = ?, jam_kerja = ?, status = 'keluar'
                                WHERE id = ?
                            """, (current_time, jam_kerja, record_id))
                            response_message = f"Checkout recorded for {employee_name}"
                        else:
                            return jsonify({
                                "message": "Minimum working time (10 minutes) not reached",
                                "status": "early_checkout"
                            }), 400
                    
                    # Prevent duplicate check-in
                    elif status == 'masuk' and current_status == 'masuk' and jam_keluar is None:
                        return jsonify({
                            "message": f"{employee_name} already checked in today",
                            "status": "already_checked_in"
                        }), 200
                        
                    # Allow new check-in if previous session was checked out
                    elif status == 'masuk' and (current_status == 'keluar' or jam_keluar is not None):
                        cursor.execute("""
                            INSERT INTO attendance (
                                employee_name, date, jam_masuk, image_capture, status
                            )
                            VALUES (?, ?, ?, ?, ?)
                        """, (employee_name, current_date, current_time, image_capture, status))
                        response_message = f"New check-in recorded for {employee_name}"
                        
                elif status == 'masuk':  # First check-in of the day
                    cursor.execute("""
                        INSERT INTO attendance (
                            employee_name, date, jam_masuk, image_capture, status
                        )
                        VALUES (?, ?, ?, ?, ?)
                    """, (employee_name, current_date, current_time, image_capture, status))
                    response_message = f"First check-in recorded for {employee_name}"

            conn.commit()
            
            # Determine time period
            time_period = get_time_period()
            
            response_data = {
                "message": response_message if 'response_message' in locals() else f"Attendance recorded for {employee_name}",
                "date": current_date,
                "time": current_time,
                "status": status,
                "period": time_period
            }
            
            if is_late and status == 'masuk':
                response_data["warning"] = "Late arrival detected"
                
            return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.route('/attendance-records', methods=['GET'])
def get_attendance_records():
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Ambil data presensi
            cursor.execute("""
                SELECT id, employee_name, date, jam_masuk, jam_keluar, jam_kerja, status, image_capture
                FROM attendance 
                ORDER BY date DESC, jam_masuk DESC
            """)

            records = cursor.fetchall()

            attendance_list = []
            for record in records:
                attendance_list.append({
                    'id': record[0],
                    'employee_name': record[1],
                    'date': record[2],
                    'jam_masuk': record[3],
                    'jam_keluar': record[4],
                    'jam_kerja': record[5],
                    'status': record[6],
                    'image_capture': record[7]
                })

            return jsonify(attendance_list), 200

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

#endpoint data presensi per karyawan
@app.route('/attendance-records/<employee_name>', methods=['GET'])
def get_employee_attendance(employee_name):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, date, jam_masuk, jam_keluar, jam_kerja, status, image_capture 
                FROM attendance 
                WHERE employee_name = ? 
                ORDER BY date DESC, jam_masuk DESC
            """, (employee_name,))

            records = cursor.fetchall()

            attendance_list = []
            for record in records:
                attendance_list.append({
                    'id': record[0],
                    'date': record[1],
                    'jam_masuk': record[2],
                    'jam_keluar': record[3],
                    'jam_kerja': record[4],
                    'status': record[5],
                    'image_capture': record[6]
                })

            return jsonify(attendance_list), 200

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

# Update employee in employee_info and employees table
@app.route('/update-employee/<int:employee_id>', methods=['PUT'])
def update_employee(employee_id):
    data = request.json
    name = data.get('name')
    position = data.get('position')
    image_base64 = data.get('image_base64', None)  # Optional field

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Update employee_info
            if image_base64:
                cursor.execute("""
                    UPDATE employee_info SET name = ?, position = ?, image_base64 = ?
                    WHERE employee_id = ?
                """, (name, position, image_base64, employee_id))
            else:
                cursor.execute("""
                    UPDATE employee_info SET name = ?, position = ?
                    WHERE employee_id = ?
                """, (name, position, employee_id))

            # Update employees (embedding table)
            cursor.execute("""
                UPDATE employees SET name = ?, position = ?
                WHERE id = ?
            """, (name, position, employee_id))
            
            conn.commit()
            return jsonify({"message": "Karyawan berhasil diperbarui!"}), 200
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

# Delete employee from both tables
@app.route('/delete-employee/<int:employee_id>', methods=['DELETE'])
def delete_employee(employee_id):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Delete from employee_info
            cursor.execute("DELETE FROM employee_info WHERE employee_id = ?", (employee_id,))

            # Delete from employees (embedding table)
            cursor.execute("DELETE FROM employees WHERE id = ?", (employee_id,))

            conn.commit()
            return jsonify({"message": "Karyawan berhasil dihapus!"}), 200
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    init_db()  # Initialize the database
    print("Akses aplikasi di http://0.0.0.0:5000 atau http://<IP-Address>:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
    

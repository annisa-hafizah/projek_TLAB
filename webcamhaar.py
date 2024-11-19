import cv2
import base64
import requests
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from gtts import gTTS
import os
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import hashlib
import threading
from queue import Queue
import subprocess

load_dotenv()

# Replace hardcoded values with environment variables
MQTT_BROKER = os.getenv('MQTT_BROKER')
MQTT_PORT = int(os.getenv('MQTT_PORT'))
MQTT_TOPIC = os.getenv('MQTT_TOPIC')
MQTT_USER = os.getenv('MQTT_USERNAME')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
SERVER_URL = os.getenv('BACKEND_SERVER_URL')
CAMERA_CONFIG = os.getenv('CAMERA_CONFIGURE')
FACE_CASCADE = os.getenv('FACE_CASCADE_PATH')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MQTT setup
MQTT_BROKER = MQTT_BROKER
MQTT_PORT = MQTT_PORT
MQTT_TOPIC = MQTT_TOPIC
MQTT_USER = MQTT_USER
MQTT_PASSWORD = MQTT_PASSWORD
mqtt_client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("Connected to MQTT broker")
    else:
        logger.error("Failed to connect to MQTT broker, Return code %d", rc)

mqtt_client.on_connect = on_connect
# mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()


    
def get_time_period():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "pagi"
    elif 12 <= hour < 15:
        return "siang"
    elif 15 <= hour < 18:
        return "sore"
    else:
        return "malam"

def get_last_attendance_from_api(employee_name):
    """
    Fetch the latest check-in and check-out times for the given employee from the API.
    
    Args:
        employee_name (str): Name of the employee to fetch attendance for. Can include 'Unknown_' prefix
        
    Returns:
        dict: Dictionary containing attendance details with keys:
            - check_in_time (str): Time of check-in in HH:MM:SS format
            - check_out_time (str): Time of check-out in HH:MM:SS format
            - status (str): Current status ('masuk' or 'keluar')
            - date (str): Date in YYYY-MM-DD format
            - is_unknown (bool): True if this is an unknown person
        None: If no records found or error occurs
    """
    try:
        # Fetch attendance records from API
        response = requests.get(
            f"{SERVER_URL}/attendance-records",
            timeout=8
        )
        response.raise_for_status()
        
        attendance_records = response.json()
        current_date = datetime.now().date()
        
        # Debug log untuk melihat response dari API
        logger.debug(f"API Response for {employee_name}: {attendance_records}")
        
        # Tentukan apakah ini unknown person
        is_unknown = employee_name.startswith('Unknown_')
        
        # Filter records untuk karyawan atau unknown person
        employee_records = [
            record for record in attendance_records
            if record['employee_name'] == employee_name
        ]
        
        if not employee_records:
            logger.info(f"No attendance records found for: {employee_name}")
            return None
            
        latest_record = employee_records[0]
        logger.debug(f"Latest record found: {latest_record}")
        
        # Pastikan date dan jam_masuk ada
        if 'date' in latest_record and latest_record['date']:
            try:
                # Parse tanggal dari field date
                record_date = datetime.strptime(latest_record['date'], "%Y-%m-%d").date()
                
                # Parse jam masuk dan keluar
                check_in_time = latest_record.get('jam_masuk')
                check_out_time = latest_record.get('jam_keluar')
                
                # Untuk unknown person, selalu izinkan check-in baru setelah interval tertentu
                if is_unknown and check_in_time:
                    # Combine date and time untuk cek interval
                    last_check_in = datetime.combine(
                        record_date,
                        datetime.strptime(check_in_time, "%H:%M:%S").time()
                    )
                    time_since_last_record = datetime.now() - last_check_in
                    # Jika sudah lebih dari 5 menit, izinkan record baru
                    if time_since_last_record.total_seconds() > 300:  # 5 menit
                        logger.info(f"Allowing new record for unknown person after 5 minutes")
                        return None
                else:
                    # Untuk karyawan terdaftar, cek tanggal seperti biasa
                    if record_date != current_date:
                        logger.info(f"Record date {record_date} differs from current date {current_date}")
                        return None
                
                # Tentukan status
                # Untuk unknown person, selalu set 'keluar' setelah deteksi
                status = latest_record.get('status', 'keluar')
                if is_unknown:
                    status = 'keluar'
                elif check_in_time and not check_out_time:
                    status = 'masuk'
                elif check_out_time:
                    status = 'keluar'
                
                logger.info(f"Successfully retrieved attendance for {employee_name}: "
                          f"check_in={check_in_time}, check_out={check_out_time}, "
                          f"status={status}, is_unknown={is_unknown}")
                
                return {
                    "check_in_time": check_in_time,
                    "check_out_time": check_out_time,
                    "status": status,
                    "date": record_date.strftime("%Y-%m-%d"),
                    "is_unknown": is_unknown
                }
                
            except ValueError as e:
                logger.error(f"Error parsing date/time from record: {e}")
                logger.debug(f"Problematic record format: {latest_record}")
                return None
        else:
            logger.warning(f"Date not found for {employee_name}, record: {latest_record}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Error processing attendance data: {e}")
        logger.debug(f"Problematic data: {attendance_records if 'attendance_records' in locals() else 'No data'}")
        return None

def generate_face_hash(face_image_base64):
    """Generate a unique hash for the face image."""
    return hashlib.md5(face_image_base64.encode()).hexdigest()

class AsyncSpeaker:
    def __init__(self, max_queue_size=2):
        self.speech_queue = Queue(maxsize=max_queue_size)  # Limit queue size
        self.current_process = None
        self.speaker_thread = None
        self.is_running = False
        self.audio_cache = {}  # Cache for frequently used messages
        self.cache_lock = threading.Lock()
        self.start_speaker_thread()

    def start_speaker_thread(self):
        """Start the speaker thread if not already running"""
        if not self.speaker_thread or not self.speaker_thread.is_alive():
            self.is_running = True
            self.speaker_thread = threading.Thread(target=self._process_speech_queue)
            self.speaker_thread.daemon = True
            self.speaker_thread.start()

    def speak_text_async(self, text):
        """Add text to speech queue if queue is not full"""
        try:
            self.speech_queue.put_nowait(text)  # Non-blocking put
        except Queue.Full:
            logging.warning("Speech queue is full, skipping message")

    def _get_audio_file(self, text):
        """Get audio file from cache or generate new one"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        with self.cache_lock:
            if text_hash in self.audio_cache:
                return self.audio_cache[text_hash]
            
            filename = f"greeting_{text_hash}.mp3"
            if not os.path.exists(filename):
                tts = gTTS(text=text, lang='id')
                tts.save(filename)
            
            self.audio_cache[text_hash] = filename
            # Keep cache size limited
            if len(self.audio_cache) > 10:  # Keep only 10 most recent messages
                oldest_key = next(iter(self.audio_cache))
                oldest_file = self.audio_cache.pop(oldest_key)
                try:
                    if os.path.exists(oldest_file):
                        os.remove(oldest_file)
                except OSError:
                    pass
                    
            return filename

    def _process_speech_queue(self):
        """Process speech queue in background"""
        while self.is_running:
            try:
                if not self.speech_queue.empty():
                    text = self.speech_queue.get(timeout=1)
                    
                    try:
                        filename = self._get_audio_file(text)
                        
                        # Stop current speech if playing
                        if self.current_process is not None:
                            try:
                                self.current_process.terminate()
                                self.current_process.wait()
                            except:
                                pass

                        # Play new speech
                        self.current_process = subprocess.Popen(
                            ['mpg321', '-q', filename],  # -q for quiet mode
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        self.current_process.wait()
                        
                    except Exception as e:
                        logging.error(f"Error playing speech: {e}")
                        
                else:
                    # Sleep to prevent CPU spinning when queue is empty
                    time.sleep(0.1)
                    
            except Exception as e:
                logging.error(f"Error in speech thread: {e}")
                time.sleep(0.1)  # Prevent rapid spinning on error

    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.current_process:
            self.current_process.terminate()
        
        # Cleanup cached files
        with self.cache_lock:
            for filename in self.audio_cache.values():
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                except OSError:
                    pass
        
class FaceDetectionSystem:
    def __init__(self):
        self.unknown_faces = {}
        self.SERVER_URL = SERVER_URL
        self.CAPTURE_INTERVAL = 6
        self.CHECKOUT_INTERVAL = 60
        self.UNKNOWN_RECORD_INTERVAL = 300
        self.FRAME_SKIP = 3
        self.DETECTION_SCALE = 0.5
        self.attendance_records = {}
        self.cap = cv2.VideoCapture(CAMERA_CONFIG)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.frame_count = 0
        self.last_capture_time = time.time()
        self.speaker = AsyncSpeaker()

        # Optimize camera capture for Jetson
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize frame buffer
        
        # Set lower resolution if needed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _handle_unknown_person(self, face_data):
        """Handle detection and recording of unknown persons."""
        current_time = time.time()
        face_hash = generate_face_hash(face_data)  # Generate unique hash for the face
        
        # Check if we've seen this face recently
        if face_hash in self.unknown_faces:
            last_detection = self.unknown_faces[face_hash]
            time_elapsed = current_time - last_detection
            
            if time_elapsed < 300:  # 5 minutes (300 seconds)
                logger.info(f"Skipping capture for {face_hash[:8]} because it was detected {time_elapsed:.1f} seconds ago.")
                return  # Skip further processing if it's too soon
        else:
            logger.info(f"First time seeing unknown face {face_hash[:8]}, proceeding to capture.")

        # Proceed to capture and record attendance for the unknown person
        unknown_id = f"Unknown Person"
        
        try:
            # Post the unknown person's attendance
            response = requests.post(
                f"{self.SERVER_URL}/record-attendance",
                json={
                    "name": unknown_id,
                    "image": face_data,
                    "status": "masuk",
                },
                timeout=8
            )
            
            if response.status_code == 200:
                # Successfully recorded unknown person, update detection time
                self.unknown_faces[face_hash] = current_time
                logger.info(f"Successfully recorded unknown person: {unknown_id}")
                
                # MQTT and speech for unknown person
                # mqtt_client.publish(MQTT_TOPIC, "Gate remains open for unknown person")
                # self.speaker.speak_text_async(f"Selamat datang di ti leb")
            else:
                logger.error(f"Failed to record unknown person: {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to record unknown person: {e}")

    def _handle_face_recognition(self, face_data):
        try:
            response = requests.post(
                f"{self.SERVER_URL}/identify-employee",
                json={"image": face_data},
                headers={'Content-Type': 'application/json'},
                timeout=8
            )
            
            if response.status_code == 200:
                data = response.json()
                employee_name = data['name']
                
                if employee_name == "Unknown Person":
                    self._handle_unknown_person(face_data)
                    logger.info(f"Unknown person detected, sending for recording.")
                    return
                
                # Rest of the code for handling known employees remains the same
                current_time = time.time()
                last_attendance = get_last_attendance_from_api(employee_name)
                current_date = datetime.now().date()
                
                if employee_name not in self.attendance_records:
                    self.attendance_records[employee_name] = {
                        "status": "keluar" if not last_attendance else last_attendance["status"],
                        "check_in_time": None if not last_attendance else last_attendance["check_in_time"],
                        "last_detection": 0,
                        "last_checkout_time": None
                    }

                record = self.attendance_records[employee_name]

                if last_attendance:
                    record["status"] = last_attendance["status"]
                    record["check_in_time"] = last_attendance["check_in_time"]
                    if last_attendance["check_out_time"]:
                        record["last_checkout_time"] = datetime.strptime(
                            f"{last_attendance['date']} {last_attendance['check_out_time']}", 
                            "%Y-%m-%d %H:%M:%S"
                        )

                if current_time - record["last_detection"] >= self.CAPTURE_INTERVAL:
                    record["last_detection"] = current_time

                    if record["status"] == "keluar":
                        self._record_attendance(employee_name, face_data, "masuk")
                        record["status"] = "masuk"
                        record["check_in_time"] = datetime.now().strftime("%H:%M:%S")
                        logger.info(f"{employee_name} checked in")
                        mqtt_client.publish(MQTT_TOPIC, f"{employee_name} checked in")

                    elif record["status"] == "masuk":
                        if record["check_in_time"]:
                            check_in_time = datetime.strptime(record["check_in_time"], "%H:%M:%S").time()
                            check_in_datetime = datetime.combine(current_date, check_in_time)
                            time_since_checkin = (datetime.now() - check_in_datetime).total_seconds()
                            
                            if time_since_checkin >= self.CHECKOUT_INTERVAL:
                                self._record_attendance(employee_name, face_data, "keluar")
                                record["status"] = "keluar"
                                record["last_checkout_time"] = datetime.now()
                                record["check_in_time"] = None
                                logger.info(f"{employee_name} checked out after {time_since_checkin/60:.1f} minutes")
                                mqtt_client.publish(MQTT_TOPIC, f"{employee_name} checked out")
                        else:
                            logger.warning(f"Check-in time not found for {employee_name}, updating status to checked out")
                            self._record_attendance(employee_name, face_data, "keluar")
                            record["status"] = "keluar"
                            record["last_checkout_time"] = datetime.now()

        except requests.exceptions.RequestException as e:
            logger.error(f"Server communication error: {e}")

    def _record_attendance(self, employee_name, image_base64, status, is_unknown=False):
        try:
            response = requests.post(
                f"{self.SERVER_URL}/record-attendance",
                json={
                    "name": employee_name,
                    "image": image_base64,
                    "status": status
                },
                timeout=8
            )
            
            if response.status_code == 200:
                data = response.json()
                time_period = data.get('period', get_time_period())
                
                if "warning" in data:
                    logger.warning(f"{employee_name} {status} - {time_period} - {data['warning']}")
                else:
                    logger.info(f"{employee_name} {status} - {time_period}")
                
                if not is_unknown:
                    if status == "masuk":
                        self.speaker.speak_text_async(
                            f"Selamat {time_period} {employee_name}, selamat datang di ti leb, silakan {status}"
                        )
                    else:
                        self.speaker.speak_text_async(
                            f"sampai jumpa {employee_name}, hati-hati di jalan"
                        )
                                   
            elif response.status_code == 400 and "early_checkout" in response.json().get("status", ""):
                logger.warning(f"Early checkout attempt for {employee_name} - Minimum working time not reached")
            else:
                logger.error(f"Failed to record attendance: {response.text}")

        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")

    def cleanup(self):
        """Cleanup resources"""
        self.speaker.cleanup()
        self.cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        self.frame_count += 1
        if self.frame_count % self.FRAME_SKIP != 0:
            return frame

        height, width = frame.shape[:2]
        small_frame = cv2.resize(frame, (int(width * self.DETECTION_SCALE), 
                                       int(height * self.DETECTION_SCALE)))
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, 
                                                 minNeighbors=5)

        faces = [(int(x/self.DETECTION_SCALE), int(y/self.DETECTION_SCALE),
                 int(w/self.DETECTION_SCALE), int(h/self.DETECTION_SCALE)) 
                 for (x,y,w,h) in faces]

        current_time = time.time()
        if faces and (current_time - self.last_capture_time >= self.CAPTURE_INTERVAL):
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            
            _, buffer = cv2.imencode('.jpg', face)
            face_base64 = base64.b64encode(buffer).decode('utf-8')
            logger.info(f"Captured face, size: {len(face_base64)} bytes")

            self._handle_face_recognition(face_base64)
            self.last_capture_time = current_time

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return frame

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break

                processed_frame = self.process_frame(frame)
                cv2.imshow('Face Detection', processed_frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    system = FaceDetectionSystem()
    try:
        system.run()
    finally:
        system.cleanup()
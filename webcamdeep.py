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
from queue import Queue, Empty
import subprocess
from logging.handlers import TimedRotatingFileHandler
import os
from datetime import datetime

# Add CUDA-specific imports
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Enable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

# Try importing TensorRT optimized DeepFace if available
try:
    from deepface import DeepFace
    import tensorflow as tf
    
    # Configure TensorFlow for Jetson
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
except ImportError as e:
    print(f"Error importing DeepFace or TensorFlow: {e}")

load_dotenv()

# Configure logging with reduced verbosity
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Environment variables
MQTT_BROKER = os.getenv('MQTT_BROKER')
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_TOPIC = os.getenv('MQTT_TOPIC')
MQTT_USER = os.getenv('MQTT_USERNAME')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
SERVER_URL = os.getenv('BACKEND_SERVER_URL')
CAMERA_CONFIG = os.getenv('CAMERA_CONFIGURE', 0)

# MQTT setup with error handling
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # Use VERSION2

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logger.info("Connected to MQTT broker")
        mqtt_client.subscribe(MQTT_TOPIC)
    else:
        logger.error(f"Failed to connect to MQTT broker, Return code {rc}")

mqtt_client.on_connect = on_connect

# Try to connect to MQTT broker with error handling
try:
    if MQTT_USER and MQTT_PASSWORD:
        mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
except Exception as e:
    logger.error(f"MQTT Connection failed: {e}")
    # Continue without MQTT if connection fails

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
    
class LoggerSetup:
    @staticmethod
    def setup_logger():
        # Create logs directory if it doesn't exist
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Generate log filename with current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        log_file = os.path.join(log_dir, 'face_detection.log')

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Setup file handler with rotation
        file_handler = TimedRotatingFileHandler(
            log_file,
            when='midnight',    # Rotate at midnight
            interval=1,         # Rotate every 1 day
            backupCount=2,     # Keep 30 days of logs
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)

        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Setup logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers if any
        logger.handlers = []
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger
    
class MQTTHandler:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.connected = False
        self.reconnect_thread = None
        self.message_queue = Queue()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to MQTT broker")
            self.connected = True
            self.process_queued_messages()
        else:
            logger.error(f"Failed to connect to MQTT broker, return code: {rc}")
            self.connected = False

    def on_disconnect(self, client, userdata, rc):
        logger.warning("Disconnected from MQTT broker")
        self.connected = False
        self.start_reconnection()

    def start_reconnection(self):
        if not self.reconnect_thread or not self.reconnect_thread.is_alive():
            self.reconnect_thread = threading.Thread(target=self.reconnect_loop)
            self.reconnect_thread.daemon = True
            self.reconnect_thread.start()

    def reconnect_loop(self):
        while not self.connected:
            try:
                logger.info("Attempting to reconnect to MQTT broker...")
                self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
                break
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                time.sleep(5)

    def publish(self, topic, message):
        try:
            if self.connected:
                self.client.publish(topic, message)
            else:
                self.message_queue.put((topic, message))
        except Exception as e:
            logger.error(f"Error publishing MQTT message: {e}")
            self.message_queue.put((topic, message))

    def process_queued_messages(self):
        while not self.message_queue.empty():
            topic, message = self.message_queue.get()
            try:
                self.client.publish(topic, message)
            except Exception as e:
                logger.error(f"Error publishing queued message: {e}")
                self.message_queue.put((topic, message))

    def start(self):
        try:
            if MQTT_USER and MQTT_PASSWORD:
                self.client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Failed to start MQTT client: {e}")
            self.connected = False

class CameraHandler:
    def __init__(self, camera_config):
        self.camera_config = camera_config
        self.cap = None
        self.frame_buffer = Queue(maxsize=2)  # Reduced buffer size
        self.connect()

    def connect(self):
        try:
            gst_str = (
                f"nvarguscamerasrc ! "
                f"video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=15/1 ! "
                f"nvvidconv ! "
                f"video/x-raw, width=640, height=480, format=(string)BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=(string)BGR ! "
                f"appsink"
            )
            
            # Try using GStreamer pipeline first (for Jetson camera)
            self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
            
            # If GStreamer fails, fall back to regular camera
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_config)
                if not self.cap.isOpened():
                    raise Exception("Failed to open camera")
                
                # Optimize camera settings
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 15)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            logger.info("Camera connected successfully")
            return True
        except Exception as e:
            logger.error(f"Camera connection failed: {e}")
            return False

    def read_frame(self):
        if not self.cap or not self.cap.isOpened():
            if not self.connect():
                return False, None
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame, attempting to reconnect...")
                if self.connect():
                    return self.cap.read()
                return False, None
            
            # Reduce frame size for processing
            frame = cv2.resize(frame, (640, 480))
            return ret, frame
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None

    def release(self):
        if self.cap:
            self.cap.release()
    

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
        self.logger = LoggerSetup.setup_logger()
        self.logger.info("Initializing Face Detection System...")

        # Initialize with optimized settings
        self.SERVER_URL = os.getenv('BACKEND_SERVER_URL', "http://localhost:8000")
        self.unknown_faces = {}
        self.attendance_records = {}
        self.frame_count = 0
        self.last_capture_time = time.time()
        self.speaker = AsyncSpeaker()
        self.mqtt_handler = MQTTHandler()
        self.camera_handler = CameraHandler(CAMERA_CONFIG)

        # Optimized performance settings for Jetson
        self.CAPTURE_INTERVAL = 10  # Increased to reduce processing load
        self.CHECKOUT_INTERVAL = 60
        self.FRAME_SKIP = 5  # Process fewer frames
        self.DETECTION_SCALE = 0.25  # Reduce image size more
        
        # Initialize DeepFace with optimized settings
        self.detector_backend = "ssd"  # Faster than retinaface for Jetson
        self.detection_model = None
        self.initialize_detection_model()
        
        # Start MQTT handler
        self.mqtt_handler.start()
        self.logger.info("Face Detection System initialized successfully")

    def initialize_detection_model(self):
        try:
            # Pre-load the model to avoid repeated loading
            self.detection_model = DeepFace.build_model(self.detector_backend)
            logger.info("Face detection model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading detection model: {e}")

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
        """Handle face recognition with proper error handling"""
        try:
            if not self.SERVER_URL:
                logger.error("SERVER_URL not available")
                return

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
                    logger.info("Unknown person detected, sending for recording.")
                    return
                    
                # Process known employee
                self._process_known_employee(employee_name, face_data)
                
            else:
                logger.error(f"Server returned status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Server communication error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in face recognition: {e}")

    def _process_known_employee(self, employee_name, face_data):
        """Process attendance for known employees"""
        try:
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
            
            # Update attendance record
            if current_time - record["last_detection"] >= self.CAPTURE_INTERVAL:
                self._update_attendance_record(employee_name, face_data, record, current_date)
                
        except Exception as e:
            logger.error(f"Error processing known employee {employee_name}: {e}")

    def _update_attendance_record(self, employee_name, face_data, record, current_date):
        """Update attendance record with proper error handling"""
        try:
            record["last_detection"] = time.time()

            if record["status"] == "keluar":
                self._record_attendance(employee_name, face_data, "masuk")
                record["status"] = "masuk"
                record["check_in_time"] = datetime.now().strftime("%H:%M:%S")
                logger.info(f"{employee_name} checked in")
                self.mqtt_handler.publish(MQTT_TOPIC, f"{employee_name} checked in")

            elif record["status"] == "masuk" and record["check_in_time"]:
                check_in_time = datetime.strptime(record["check_in_time"], "%H:%M:%S").time()
                check_in_datetime = datetime.combine(current_date, check_in_time)
                time_since_checkin = (datetime.now() - check_in_datetime).total_seconds()
                
                if time_since_checkin >= self.CHECKOUT_INTERVAL:
                    self._record_attendance(employee_name, face_data, "keluar")
                    record["status"] = "keluar"
                    record["last_checkout_time"] = datetime.now()
                    record["check_in_time"] = None
                    logger.info(f"{employee_name} checked out after {time_since_checkin/60:.1f} minutes")
                    self.mqtt_handler.publish(MQTT_TOPIC, f"{employee_name} checked out")
                    
        except Exception as e:
            logger.error(f"Error updating attendance record for {employee_name}: {e}")

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
                time_period = get_time_period()#data.get('period')
                
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

    def _process_frame_thread(self):
        """Thread for processing frames"""
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    break
                    
                processed = self._process_single_frame(frame)
                
                try:
                    self.result_queue.put_nowait(processed)
                except:
                    pass  # Skip if queue is full
                    
            except Empty:
                continue  # Continue if queue is empty
            except Exception as e:
                logger.error(f"Error in processing thread: {e}")
                continue

    def _process_single_frame(self, frame):
        """Process single frame for face detection using DeepFace"""
        try:
            # Detect faces using DeepFace
            face_objs = DeepFace.extract_faces(
                frame,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )

            faces = []
            for face_obj in face_objs:
                facial_area = face_obj['facial_area']
                x = facial_area['x']
                y = facial_area['y']
                w = facial_area['w']
                h = facial_area['h']
                faces.append((x, y, w, h))

            # Draw rectangles with celadon green color (RGB: 178, 255, 178)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (178, 255, 178), 2)

            if faces:
                current_time = time.time()
                if current_time - self.last_capture_time >= self.CAPTURE_INTERVAL:
                    x, y, w, h = faces[0]
                    face = frame[y:y+h, x:x+w]
                    
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    _, buffer = cv2.imencode('.jpg', face, encode_param)
                    face_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    threading.Thread(
                        target=self._handle_face_recognition,
                        args=(face_base64,),
                        daemon=True
                    ).start()
                    
                    self.last_capture_time = current_time

            return frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame

    def process_frame(self, frame):
        try:
            self.frame_count += 1
            if self.frame_count % self.FRAME_SKIP != 0:
                return frame

            # Optimize frame processing
            height, width = frame.shape[:2]
            small_frame = cv2.resize(frame, 
                                   (int(width * self.DETECTION_SCALE), 
                                    int(height * self.DETECTION_SCALE)),
                                   interpolation=cv2.INTER_AREA)

            # Convert to RGB (DeepFace expects RGB)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect faces using pre-loaded model
            try:
                face_objs = DeepFace.extract_faces(
                    rgb_small_frame,
                    detector_backend=self.detector_backend,
                    enforce_detection=False,
                    align=False  # Disable alignment for speed
                )

                faces = []
                for face_obj in face_objs:
                    facial_area = face_obj['facial_area']
                    x = int(facial_area['x'] / self.DETECTION_SCALE)
                    y = int(facial_area['y'] / self.DETECTION_SCALE)
                    w = int(facial_area['w'] / self.DETECTION_SCALE)
                    h = int(facial_area['h'] / self.DETECTION_SCALE)
                    faces.append((x, y, w, h))

                current_time = time.time()
                if faces and (current_time - self.last_capture_time >= self.CAPTURE_INTERVAL):
                    x, y, w, h = faces[0]
                    face = frame[y:y+h, x:x+w]
                    
                    # Optimize image encoding
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # Reduced quality
                    _, buffer = cv2.imencode('.jpg', face, encode_param)
                    face_base64 = base64.b64encode(buffer).decode('utf-8')

                    # Use thread pool instead of creating new threads
                    threading.Thread(
                        target=self._handle_face_recognition,
                        args=(face_base64,),
                        daemon=True
                    ).start()
                    
                    self.last_capture_time = current_time

                # Only draw rectangles if display window is visible
                if cv2.getWindowProperty('Face Detection', cv2.WND_PROP_VISIBLE) >= 0:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (178, 255, 178), 2)

            except Exception as e:
                logger.debug(f"No faces detected in frame: {e}")

            return frame

        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
            return frame

    def run(self):
        try:
            logger.info("Starting face detection system...")
            while True:
                ret, frame = self.camera_handler.read_frame()
                if not ret:
                    logger.warning("No frame available, waiting...")
                    time.sleep(1)
                    continue

                processed_frame = self.process_frame(frame)
                
                try:
                    # Use NVIDIA hardware acceleration for display
                    cv2.imshow('Face Detection', processed_frame)
                except Exception as e:
                    logger.error(f"Error displaying frame: {e}")

                if cv2.waitKey(1) & 0xFF == 27:
                    break

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        self.camera_handler.release()
        cv2.destroyAllWindows()
        self.mqtt_handler.client.loop_stop()
        if hasattr(self, 'speaker'):
            self.speaker.cleanup()

def main():
    # Set process priority
    try:
        os.nice(-10)  # Higher priority for the process
    except:
        pass

    logger = LoggerSetup.setup_logger()
    logger.info("Starting Face Detection System...")

    try:
        system = FaceDetectionSystem()
        system.run()
    except Exception as e:
        logger.error(f"Critical error in main: {e}", exc_info=True)
    finally:
        logger.info("Shutting down Face Detection System...")

if __name__ == "__main__":
    main()
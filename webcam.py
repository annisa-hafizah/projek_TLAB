import cv2
import base64
import requests
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from gtts import gTTS
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def speak_text(text):
    tts = gTTS(text=text, lang='id')
    tts.save("greeting.mp3")
    os.system("mpg321 greeting.mp3")
    
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
        
class FaceDetectionSystem:
    def __init__(self):
        self.SERVER_URL = "http://172.254.3.21:5000"
        self.CAPTURE_INTERVAL = 5
        self.CHECKOUT_INTERVAL = 600  # 10 minutes in seconds
        self.FRAME_SKIP = 2
        self.DETECTION_SCALE = 0.5
        self.unknown_cooldown = 300
        self.last_unknown_detection = 0
        
        # Enhanced attendance tracking
        self.attendance_records = {}  # Track employee attendance status
        self.last_detection_time = {}  # Track last detection time for each employee
        
        self.cap = cv2.VideoCapture('http://172.254.0.124:2000/video')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.frame_count = 0
        self.last_capture_time = time.time()

    def _handle_face_recognition(self, face_data):
        try:
            response = requests.post(
                f"{self.SERVER_URL}/identify-employee",
                json={"image": face_data},
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                employee_name = data['name']
                current_time = time.time()
                
                # Handle Unknown Person
                if employee_name == "Unknown Person":
                    if current_time - self.last_unknown_detection >= self.unknown_cooldown:
                        self._record_attendance(employee_name, face_data, "masuk")
                        self.last_unknown_detection = current_time
                    return

                # Initialize employee record if not exists
                if employee_name not in self.attendance_records:
                    self.attendance_records[employee_name] = {
                        "status": "keluar",
                        "check_in_time": None,
                        "last_detection": 0
                    }

                record = self.attendance_records[employee_name]
                
                # Handle new detection
                if current_time - record["last_detection"] >= self.CAPTURE_INTERVAL:
                    record["last_detection"] = current_time
                    
                    # Handle check-in
                    if record["status"] == "keluar":
                        self._record_attendance(employee_name, face_data, "masuk")
                        record["status"] = "masuk"
                        record["check_in_time"] = current_time
                        logger.info(f"{employee_name} checked in")
                    
                    # Handle check-out
                    elif record["status"] == "masuk":
                        time_since_checkin = current_time - record["check_in_time"]
                        
                        # Only attempt checkout if minimum time has passed
                        if time_since_checkin >= self.CHECKOUT_INTERVAL:
                            self._record_attendance(employee_name, face_data, "keluar")
                            record["status"] = "keluar"
                            record["check_in_time"] = None
                            logger.info(f"{employee_name} checked out after {time_since_checkin/60:.1f} minutes")

        except requests.exceptions.RequestException as e:
            logger.error(f"Server communication error: {e}")

    def _record_attendance(self, employee_name, image_base64, status):
        try:
            response = requests.post(
                f"{self.SERVER_URL}/record-attendance",
                json={
                    "name": employee_name,
                    "image": image_base64,
                    "status": status
                },
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                time_period = data.get('period', get_time_period())
                
                if "warning" in data:
                    logger.warning(f"{employee_name} {status} - {time_period} - {data['warning']}")
                else:
                    logger.info(f"{employee_name} {status} - {time_period}")
                    
            elif response.status_code == 400 and "early_checkout" in response.json().get("status", ""):
                logger.warning(f"Early checkout attempt for {employee_name} - Minimum working time not reached")
            else:
                logger.error(f"Failed to record attendance: {response.text}")

        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")

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
    system.run()
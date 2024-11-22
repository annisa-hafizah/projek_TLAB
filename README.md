# projek_TLAB
spesifikasi minimum yang dibutuhkan:

untuk sistem operasi windows dan linux : python 3.10.12
- intel i3 gen 7 dual core 4 threads 3.90ghz
- ram 12gb
- Dedicated GPU
  
MQTT BROKER dijalankan di linux : mosquitto 2.0.11
- AMD Ryzen 7 3700U
- ram 8 GB
- Dedicated GPU
  
untuk menjalankan aplikasi app.py dijalankan di server dan webcamhaar.py dijalankan di client(jetson) dan untuk employee_registration.py dijalankan di server 

To deploy webcamdeep.py on your Jetson Nano, you'll need to:

Install required packages:

sudo apt-get update

sudo apt-get install python3-pip python3-dev python3-opencv

sudo apt-get install libhdf5-serial-dev hdf5-tools

sudo apt-get install libjpeg-dev zlib1g-dev

sudo apt-get install gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly

Install Python packages:

pip3 install numpy tensorflow-gpu==2.10.0

pip3 install deepface opencv-python paho-mqtt python-dotenv gtts

Set up your environment variables in .env file:

MQTT_BROKER=your_broker
MQTT_PORT=1883
MQTT_TOPIC=your_topic
MQTT_USERNAME=your_username
MQTT_PASSWORD=your_password
BACKEND_SERVER_URL=http://your_server:port
CAMERA_CONFIGURE=0

Run the script:

python3 webcamdeep.py

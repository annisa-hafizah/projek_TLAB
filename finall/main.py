import usocket as socket
from machine import Pin
import network
import time  # Import time module for the delay

# Initialize relay pins
relay1 = Pin(4, Pin.OUT)  # First relay (Open gate)
relay2 = Pin(5, Pin.OUT)  # Second relay (Close gate)

# Wi-Fi credentials
ssid = 'TLab Head Office'
password = 'mudabersemangat'

# Connect to Wi-Fi
station = network.WLAN(network.STA_IF)
station.active(True)
station.connect(ssid, password)

while not station.isconnected():
    pass

print('Connection successful')
print(station.ifconfig())

# Gate state variable
gate_open = False

# Function to open the gate
def open_gate():
    global gate_open
    if not gate_open:
        relay1.value(1)  # Activate first relay to open gate
        relay2.value(0)  # Deactivate second relay (gate open state)
        gate_open = True
        # Start a timer to close the gate after 10 seconds
        time.sleep(10)  # Wait for 10 seconds
        close_gate()

# Function to close the gate
def close_gate():
    global gate_open
    if gate_open:
        relay1.value(0)  # Deactivate first relay (gate closed state)
        relay2.value(1)  # Activate second relay to close gate
        gate_open = False

# Function to serve static files
def serve_file(file_name, content_type):
    try:
        with open(file_name, 'r') as file:
            response = file.read()
        conn.send('HTTP/1.1 200 OK\n')
        conn.send(f'Content-Type: {content_type}\n')
        conn.send('Connection: close\n\n')
        conn.sendall(response)
    except Exception as e:
        conn.send('HTTP/1.1 404 Not Found\n')
        conn.send('Connection: close\n\n')
        conn.sendall("File not found")

# Setup server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 80))
s.listen(5)

while True:
    conn, addr = s.accept()
    print('Got connection from', addr)
    
    request = conn.recv(1024).decode()
    print(f"Request: {request}")

    # Handle specific URLs for opening and closing the gate
    if 'GET /open-gate' in request:
        open_gate()
        conn.send('HTTP/1.1 200 OK\n')
        conn.send('Content-Type: text/html\n')
        conn.send('Connection: close\n\n')
        conn.sendall("Gate is OPEN")
    
    elif 'GET /close-gate' in request:
        close_gate()
        conn.send('HTTP/1.1 200 OK\n')
        conn.send('Content-Type: text/html\n')
        conn.send('Connection: close\n\n')
        conn.sendall("Gate is CLOSED")
    
    # Serve HTML, CSS, or JavaScript based on request
    elif request.startswith('GET / '):
        serve_file('index.html', 'text/html')
    elif request.startswith('GET /style.css'):
        serve_file('style.css', 'text/css')
    elif request.startswith('GET /script.js'):
        serve_file('script.js', 'application/javascript')

    conn.close()

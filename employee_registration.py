import streamlit as st
import requests
import base64
import json
import cv2
import numpy as np
import pandas as pd

# Fungsi untuk mengonversi file gambar ke base64
def convert_image_to_base64(image):
    return base64.b64encode(image.read()).decode("utf-8")

# Sidebar dengan dua elemen
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Pilih Menu", ["Pendaftaran Karyawan", "List Data Karyawan"])

# Function to crop face from the image
def crop_face(image):
    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Crop the first detected face (if any)
    for (x, y, w, h) in faces:
        return image[y:y + h, x:x + w]  # Return the cropped face

    return None  # If no face is detected

# Fungsi untuk membaca gambar sebagai numpy array dari file Streamlit
def read_image(uploaded_image):
    image_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, 1)  # Decode the image as OpenCV format
    return image

# Jika pilihan di sidebar adalah "Pendaftaran Karyawan"
if menu == "Pendaftaran Karyawan":
    st.title("Pendaftaran Karyawan")

    # Input nama, posisi, dan foto karyawan
    name = st.text_input("Nama Karyawan")

    # Select box untuk posisi karyawan di PT Teknologi Kode Indonesia
    positions = [
        "President Director",
        "IT Manager",
        "Software Engineer",
        "Data Analyst",
        "IT Support",
        "Quality Assurance",
        "Product Manager",
        "HR Manager",
        "Accountant",
        "Marketing Specialist",
        "UI/UX Designer",
        "Business Analyst" 
    ]
    position = st.selectbox("Posisi Karyawan", positions)

    foto = st.camera_input("Capture Foto Karyawan")
    uploaded_image = st.file_uploader("Unggah Foto Karyawan", type=["jpg", "jpeg", "png"])

    # Tombol untuk mengirim data ke API
    if st.button("Daftarkan Karyawan"):
        if foto and uploaded_image:
            st.error("Pilih salah satu: Capture foto atau unggah foto. Tidak boleh keduanya.")
        elif name and position and (foto or uploaded_image):
            # Pilih antara foto yang diambil dari kamera atau yang diunggah
            if foto:
                image = read_image(foto)
            else:
                image = read_image(uploaded_image)

            # Crop wajah dari gambar
            cropped_face = crop_face(image)

            if cropped_face is not None:
                # Konversi hasil crop wajah ke base64
                _, buffer = cv2.imencode('.jpg', cropped_face)
                image_base64 = base64.b64encode(buffer).decode("utf-8")

                # Data JSON yang akan dikirim ke API
                data = {
                    "name": name,
                    "position": position,
                    "image": image_base64  # Gambar dalam format base64
                }

                # Kirim request POST ke API Flask
                response = requests.post("http://127.0.0.1:5000/register-employee", json=data)

                # Tampilkan respon dari API
                if response.status_code == 200:
                    st.success("Karyawan berhasil didaftarkan!")
                else:
                    st.error(f"Error: {response.text}")
            else:
                st.error("Wajah tidak terdeteksi di gambar, coba lagi.")
        else:
            st.warning("Mohon lengkapi semua data sebelum mengirim.")
            
# Jika pilihan di sidebar adalah "List Data Karyawan"
elif menu == "List Data Karyawan":
    st.title("List Data Karyawan")

    # Kirim request GET ke API untuk mengambil data karyawan
    response = requests.get("http://127.0.0.1:5000/employees")

    if response.status_code == 200:
        data = response.json()  # Data JSON dari API
        if data:
            # Konversi data ke DataFrame untuk ditampilkan dalam tabel
            df = pd.DataFrame(data)

            # Tampilkan tabel dengan nomor, nama, dan posisi
            st.table(df[["name", "posisi"]])
        else:
            st.write("Tidak ada data karyawan yang tersedia.")
    else:
        st.error(f"Gagal mengambil data karyawan. Error: {response.text}")

import streamlit as st
import requests
import base64
import json
import cv2
import numpy as np
import pandas as pd

# Function to convert file image to base64
def convert_image_to_base64(image):
    return base64.b64encode(image.read()).decode("utf-8")

# List of positions available
positions = [
    "President Director", "IT Manager", "Software Engineer", "Data Analyst", 
    "IT Support", "Quality Assurance", "Product Manager", "HR Manager", 
    "Accountant", "Marketing Specialist", "UI/UX Designer", "Business Analyst"
]

# Sidebar with three menu options
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Pilih Menu", ["Pendaftaran Karyawan", "List Data Karyawan", "Edit/Hapus Karyawan"])

# Function to crop face from the image
def crop_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        return image[y:y + h, x:x + w]  # Return the cropped face
    return None

# Function to read image as numpy array from Streamlit file input
def read_image(uploaded_image):
    image_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, 1)  # Decode the image as OpenCV format
    return image

# Employee registration
if menu == "Pendaftaran Karyawan":
    st.title("Pendaftaran Karyawan")

    # Input employee name, position, and photo
    name = st.text_input("Nama Karyawan")

    # Using the positions list defined globally
    position = st.selectbox("Posisi Karyawan", positions)

    foto = st.camera_input("Capture Foto Karyawan")
    uploaded_image = st.file_uploader("Unggah Foto Karyawan", type=["jpg", "jpeg", "png"])

    if st.button("Daftarkan Karyawan"):
        if foto and uploaded_image:
            st.error("Pilih salah satu: Capture foto atau unggah foto. Tidak boleh keduanya.")
        elif name and position and (foto or uploaded_image):
            if foto:
                image = read_image(foto)
            else:
                image = read_image(uploaded_image)

            cropped_face = crop_face(image)

            if cropped_face is not None:
                _, buffer = cv2.imencode('.jpg', cropped_face)
                image_base64 = base64.b64encode(buffer).decode("utf-8")

                data = {"name": name, "position": position, "image": image_base64}
                response = requests.post("http://127.0.0.1:5000/register-employee", json=data)

                if response.status_code == 200:
                    st.success("Karyawan berhasil didaftarkan!")
                else:
                    st.error(f"Error: {response.text}")
            else:
                st.error("Wajah tidak terdeteksi di gambar, coba lagi.")
        else:
            st.warning("Mohon lengkapi semua data sebelum mengirim.")

# List employees
elif menu == "List Data Karyawan":
    st.title("List Data Karyawan")

    response = requests.get("http://127.0.0.1:5000/employees")

    if response.status_code == 200:
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            st.table(df[["name", "position"]])
        else:
            st.write("Tidak ada data karyawan yang tersedia.")
    else:
        st.error(f"Gagal mengambil data karyawan. Error: {response.text}")

# Edit or Delete employees
elif menu == "Edit/Hapus Karyawan":
    st.title("Edit atau Hapus Data Karyawan")

    # Fetch list of employees for selection
    response = requests.get("http://127.0.0.1:5000/employees")
    if response.status_code == 200:
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            employee = st.selectbox("Pilih Karyawan untuk Diedit atau Dihapus", df["name"])
            selected_employee = df[df["name"] == employee].iloc[0]

            # Edit employee details
            st.subheader("Edit Data Karyawan")
            updated_name = st.text_input("Nama Karyawan", value=selected_employee["name"])
            updated_position = st.selectbox("Posisi Karyawan", positions, index=positions.index(selected_employee["position"]))

            if st.button("Simpan Perubahan"):
                update_data = {"name": updated_name, "position": updated_position}
                response = requests.put(f"http://127.0.0.1:5000/update-employee/{selected_employee['id']}", json=update_data)

                if response.status_code == 200:
                    st.success("Data karyawan berhasil diperbarui!")
                else:
                    st.error(f"Error: {response.text}")

            # Delete employee
            st.subheader("Hapus Karyawan")
            if st.button("Hapus Karyawan"):
                response = requests.delete(f"http://127.0.0.1:5000/delete-employee/{selected_employee['id']}")
                if response.status_code == 200:
                    st.success("Karyawan berhasil dihapus!")
                else:
                    st.error(f"Error: {response.text}")
    else:
        st.error(f"Gagal mengambil data karyawan. Error: {response.text}")
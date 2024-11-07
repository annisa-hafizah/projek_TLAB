import streamlit as st
import requests
import base64
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from datetime import datetime

# Function to convert file image to base64
def convert_image_to_base64(image):
    return base64.b64encode(image.read()).decode("utf-8")

SERVER_URL = "http://172.254.2.153:5000"

# List of positions available
positions = [
    "President Director", "IT Manager", "Software Engineer", "Data Analyst", 
    "IT Support", "Quality Assurance", "Product Manager", "HR Manager", 
    "Accountant", "Marketing Specialist", "UI/UX Designer", "Business Analyst"
]

# Sidebar with menu options
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Pilih Menu", ["Pendaftaran Karyawan", "Data Karyawan", "Edit/Hapus Karyawan","Log Attendance"])

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

    name = st.text_input("Nama Karyawan")
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
                response = requests.post(f"{SERVER_URL}/register-employee", json=data)

                if response.status_code == 200:
                    st.success("Karyawan berhasil didaftarkan!")
                else:
                    st.error(f"Error: {response.text}")
            else:
                st.error("Wajah tidak terdeteksi di gambar, coba lagi.")
        else:
            st.warning("Mohon lengkapi semua data sebelum mengirim.")

# List employees with photos
elif menu == "Data Karyawan":
    st.title("Data Karyawan")

    response = requests.get(f"{SERVER_URL}/employees-info")

    if response.status_code == 200:
        employees = response.json()
        if employees:
            # Create a list to hold the rows of our table
            table_data = []
            
            for emp in employees:
                # Convert base64 image to displayable format
                image_data = base64.b64decode(emp['image_base64'])
                img = Image.open(BytesIO(image_data))
                
                # Resize image to make it smaller in the table
                img.thumbnail((100, 100))
                
                # Convert PIL Image to bytes
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Create HTML for the image
                img_html = f'<img src="data:image/png;base64,{img_str}" style="width:100px;">'
                
                # Add this employee's data to our table
                table_data.append([emp['name'], emp['position'], img_html])
            
            # Create DataFrame
            df = pd.DataFrame(table_data, columns=["Nama", "Posisi", "Foto"])
            
            # Display the table
            st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.write("Tidak ada data karyawan yang tersedia.")
    else:
        st.error(f"Gagal mengambil data karyawan. Error: {response.text}")

# Edit or Delete employees
elif menu == "Edit/Hapus Karyawan":
    st.title("Edit atau Hapus Data Karyawan")

    response = requests.get(f"{SERVER_URL}/employees-full")
    if response.status_code == 200:
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            employee = st.selectbox("Pilih Karyawan untuk Diedit atau Dihapus", df["name"])
            selected_employee = df[df["name"] == employee].iloc[0]

            st.subheader("Edit Data Karyawan")
            updated_name = st.text_input("Nama Karyawan", value=selected_employee["name"])
            updated_position = st.selectbox("Posisi Karyawan", positions, index=positions.index(selected_employee["position"]))

            if st.button("Simpan Perubahan"):
                update_data = {"name": updated_name, "position": updated_position}
                response = requests.put(f"{SERVER_URL}/update-employee/{selected_employee['id']}", json=update_data)

                if response.status_code == 200:
                    st.success("Data karyawan berhasil diperbarui!")
                else:
                    st.error(f"Error: {response.text}")

            st.subheader("Hapus Karyawan")
            if st.button("Hapus Karyawan"):
                response = requests.delete(f"{SERVER_URL}/delete-employee/{selected_employee['id']}")
                if response.status_code == 200:
                    st.success("Karyawan berhasil dihapus!")
                else:
                    st.error(f"Error: {response.text}")

# Log Attendance Section
elif menu == "Log Attendance":
    st.title("Log Attendance Karyawan")
    
    # Tambahkan filter
    col1, col2 = st.columns(2)
    with col1:
        emp_response = requests.get(f"{SERVER_URL}/employees-full")
        if emp_response.status_code == 200:
            employees = emp_response.json()
            # Include "Unknown Person" in the employee list
            employee_names = ["Semua Karyawan", "Unknown Person"] + [emp["name"] for emp in employees]
            selected_employee = st.selectbox("Filter berdasarkan Karyawan:", employee_names)
    
    with col2:
        filter_date = st.date_input("Filter berdasarkan Tanggal:")
    
    # Debug: Cek struktur data yang diterima
    if st.button("Tampilkan Log Attendance"):
        if selected_employee == "Semua Karyawan":
            response = requests.get(f"{SERVER_URL}/attendance-records")
        elif selected_employee == "Unknown Person":
            # Fetch records for unknown individuals
            response = requests.get(f"{SERVER_URL}/attendance-records/Unknown Person")
        else:
            response = requests.get(f"{SERVER_URL}/attendance-records/{selected_employee}")
        
        if response.status_code == 200:
            attendance_data = response.json()
            
            if attendance_data:
                # Filter berdasarkan tanggal jika dipilih
                if filter_date:
                    attendance_data = [
                        record for record in attendance_data 
                        if record['date'] == filter_date.strftime('%Y-%m-%d')
                    ]
                
                if attendance_data:
                    # Create a list to hold the rows of our table
                    table_data = []
                    
                    for record in attendance_data:
                        # Convert base64 image to displayable format
                        if record.get('image_capture'):
                            try:
                                image_data = base64.b64decode(record['image_capture'])
                                img = Image.open(BytesIO(image_data))
                                
                                # Resize image to make it smaller in the table
                                img.thumbnail((100, 100))
                                
                                # Convert PIL Image to bytes
                                buffered = BytesIO()
                                img.save(buffered, format="PNG")
                                img_str = base64.b64encode(buffered.getvalue()).decode()
                                
                                # Create HTML for the image
                                img_html = f'<img src="data:image/png;base64,{img_str}" style="width:100px;">'
                            except Exception as e:
                                st.error(f"Error processing image: {e}")
                                img_html = ""
                        else:
                            img_html = ""
                        
                        # Format tanggal
                        tanggal = datetime.strptime(record['date'], '%Y-%m-%d').strftime('%d-%m-%Y')
                        
                        # Add this record's data to our table - using get() to avoid KeyError
                        table_data.append([
                            record.get('employee_name', 'Unknown Person'),  # Default to 'Unknown Person' if not found
                            tanggal,
                            record.get('jam_masuk', ''),
                            record.get('jam_keluar', ''),
                            record.get('jam_kerja', ''),
                            record.get('status', ''),
                            img_html
                        ])
                    
                    # Create DataFrame
                    df = pd.DataFrame(table_data, columns=[
                        "Nama Karyawan",
                        "Tanggal",
                        "jam_masuk",
                        "jam_keluar",
                        "jam_kerja",
                        "status",
                        "image_capture"
                    ])
                    
                    # Display the table
                    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    
                else:
                    st.info("Tidak ada data presensi untuk filter yang dipilih.")
            else:
                st.info("Belum ada data presensi.")
        else:
            st.error(f"Gagal mengambil data presensi: {response.text}")

    # Tambahkan opsi ekspor data
    if st.button("Ekspor Data ke CSV"):
        if 'df' in locals():  # Check if DataFrame exists
            # Convert DataFrame to CSV
            csv = df.to_csv(index=False).encode('utf-8')
            
            # Download button
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f'absensi_{filter_date.strftime("%Y-%m-%d") if filter_date else "all"}.csv',
                mime='text/csv',
            )

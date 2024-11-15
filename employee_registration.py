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
import calendar
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

load_dotenv()

# Konfigurasi aplikasi
SERVER_URL = os.getenv('BACKEND_SERVER_URL')
STREAMLIT_PORT = int(os.getenv('STREAMLIT_SERVER_PORT'))
STREAMLIT_ADDRESS = os.getenv('STREAMLIT_SERVER_ADDRESS', '0.0.0.0')
IMAGE_TYPES = os.getenv('IMAGE_UPLOAD_TYPES', 'jpg,jpeg,png').split(',')
DATE_FORMAT = os.getenv('DEFAULT_DATE_FORMAT')
TIME_FORMAT = os.getenv('DEFAULT_TIME_FORMAT')
FACE_CASCADE_PATH = os.getenv('FACE_CASCADE_PATH')
# Function to convert file image to base64
def convert_image_to_base64(image):
    return base64.b64encode(image.read()).decode("utf-8")

SERVER_URL = SERVER_URL

# List of positions available
positions = [
    "President Director", "IT Manager", "Software Engineer", "Data Analyst", 
    "IT Support", "Quality Assurance", "Product Manager", "HR Manager", 
    "Accountant", "Marketing Specialist", "UI/UX Designer", "Business Analyst"
]

# Sidebar with menu options
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Pilih Menu", ["Pendaftaran Karyawan", "Data Karyawan", "Edit/Hapus Karyawan", "Log Attendance", "Analisis Data"])

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

# Helper functions for Excel export
def create_excel_download(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write the main data
        df.to_excel(writer, sheet_name='Attendance', index=False)
        
        # Get the worksheet
        worksheet = writer.sheets['Attendance']
        
        # Adjust column widths
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).apply(len).max(),
                len(col)
            )
            worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2
    
    output.seek(0)
    return output.getvalue()

def get_excel_filename(selected_employee, filter_date):
    if selected_employee == "Semua Karyawan":
        employee_part = "all_employees"
    else:
        employee_part = selected_employee.replace(" ", "_")
    
    date_part = filter_date.strftime("%Y-%m-%d") if filter_date else "all_dates"
    return f'attendance_{employee_part}_{date_part}.xlsx'

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
    
    # Inisialisasi DataFrame kosong di awal
    df_excel = pd.DataFrame()
    df_display = pd.DataFrame()
    
    # Tambahkan filter
    col1, col2 = st.columns(2)
    with col1:
        emp_response = requests.get(f"{SERVER_URL}/employees-full")
        if emp_response.status_code == 200:
            employees = emp_response.json()
            employee_names = ["Semua Karyawan", "Unknown Person"] + [emp["name"] for emp in employees]
            selected_employee = st.selectbox("Filter berdasarkan Karyawan:", employee_names)
    
    with col2:
        filter_date = st.date_input("Filter berdasarkan Tanggal:")
    
    # Fetch and display data when button is clicked
    if st.button("Tampilkan Log Attendance"):
        if selected_employee == "Semua Karyawan":
            response = requests.get(f"{SERVER_URL}/attendance-records")
        elif selected_employee == "Unknown Person":
            response = requests.get(f"{SERVER_URL}/attendance-records/Unknown Person")
        else:
            response = requests.get(f"{SERVER_URL}/attendance-records/{selected_employee}")
        
        if response.status_code == 200:
            attendance_data = response.json()
            
            if attendance_data:
                if filter_date:
                    attendance_data = [
                        record for record in attendance_data 
                        if record['date'] == filter_date.strftime('%Y-%m-%d')
                    ]
                
                if attendance_data:
                    # Prepare data for both display and Excel export
                    table_data = []
                    excel_data = []  # Separate list for Excel export without images
                    
                    for record in attendance_data:
                        # Format tanggal
                        tanggal = datetime.strptime(record['date'], '%Y-%m-%d').strftime('%d-%m-%Y')
                        
                        # Prepare Excel data (without images)
                        excel_row = [
                            record.get('employee_name', 'Unknown Person'),
                            tanggal,
                            record.get('jam_masuk', ''),
                            record.get('jam_keluar', ''),
                            record.get('jam_kerja', ''),
                            record.get('status', '')
                        ]
                        excel_data.append(excel_row)
                        
                        # Prepare display data (with images)
                        if record.get('image_capture'):
                            try:
                                image_data = base64.b64decode(record['image_capture'])
                                img = Image.open(BytesIO(image_data))
                                img.thumbnail((100, 100))
                                buffered = BytesIO()
                                img.save(buffered, format="PNG")
                                img_str = base64.b64encode(buffered.getvalue()).decode()
                                img_html = f'<img src="data:image/png;base64,{img_str}" style="width:100px;">'
                            except Exception as e:
                                st.error(f"Error processing image: {e}")
                                img_html = ""
                        else:
                            img_html = ""
                        
                        display_row = excel_row.copy()
                        display_row.append(img_html)
                        table_data.append(display_row)
                    
                    # Create DataFrames for display and Excel
                    columns = [
                        "Nama Karyawan",
                        "Tanggal",
                        "Jam Masuk",
                        "Jam Keluar",
                        "Jam Kerja",
                        "Status"
                    ]
                    
                    df_excel = pd.DataFrame(excel_data, columns=columns)
                    
                    display_columns = columns + ["Foto"]
                    df_display = pd.DataFrame(table_data, columns=display_columns)
                    
                    # Display the table
                    st.write(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
                    
                else:
                    st.info("Tidak ada data presensi untuk filter yang dipilih.")
            else:
                st.info("Belum ada data presensi.")
        else:
            st.error(f"Gagal mengambil data presensi: {response.text}")

    # Add export button - only show if we have data
    if not df_excel.empty:
        st.download_button(
            label="📥 Download Excel",
            data=create_excel_download(df_excel),
            file_name=get_excel_filename(selected_employee, filter_date),
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key='download-excel'
        )


            
elif menu == "Analisis Data":
    st.title("Analisis Data Karyawan")
    
    # Fetch all employees and attendance data
    emp_response = requests.get(f"{SERVER_URL}/employees-full")
    att_response = requests.get(f"{SERVER_URL}/attendance-records")
    
    if emp_response.status_code == 200 and att_response.status_code == 200:
        employees = emp_response.json()
        attendance = att_response.json()
        
        # Convert attendance data to DataFrame
        df_attendance = pd.DataFrame(attendance)
        if not df_attendance.empty:
            df_attendance['date'] = pd.to_datetime(df_attendance['date'])
            df_attendance['year'] = df_attendance['date'].dt.year
            df_attendance['month'] = df_attendance['date'].dt.month
            df_attendance['month_name'] = df_attendance['date'].dt.strftime('%B')
            df_attendance['jam_masuk'] = df_attendance['jam_masuk']
            # 1. Grafik rerata karyawan masuk per bulan dan tahun
            st.subheader("Rerata Kehadiran Karyawan")
            
            # Group by year and month
            monthly_attendance = df_attendance.groupby(['year', 'month', 'month_name']).agg({
                'employee_name': 'count'
            }).reset_index()
            
            # Create line chart using plotly
            fig_monthly = px.line(monthly_attendance, 
                                x='month_name', 
                                y='employee_name',
                                color='year',
                                labels={'employee_name': 'Jumlah Kehadiran',
                                       'month_name': 'Bulan',
                                       'year': 'Tahun'},
                                title='Rerata Kehadiran Karyawan per Bulan')
            st.plotly_chart(fig_monthly)
            
            # 2. Grafik rata-rata durasi jam kerja
            st.subheader("Rata-rata Durasi Jam Kerja")
            
            # Convert jam_kerja to hours
            def convert_time_to_hours(time_str):
                if pd.isna(time_str):
                    return 0
                try:
                    hours = pd.Timedelta(time_str).total_seconds() / 3600
                    return round(hours, 2)
                except:
                    return 0
            
            df_attendance['working_hours'] = df_attendance['jam_kerja'].apply(convert_time_to_hours)
            
            # Calculate average working hours per employee
            avg_working_hours = df_attendance.groupby('employee_name')['working_hours'].mean().reset_index()
            avg_working_hours = avg_working_hours.sort_values('working_hours', ascending=True)
            
            # Create bar chart
            fig_hours = px.bar(avg_working_hours,
                             x='employee_name',
                             y='working_hours',
                             labels={'employee_name': 'Nama Karyawan',
                                    'working_hours': 'Rata-rata Jam Kerja'},
                             title='Rata-rata Durasi Jam Kerja per Karyawan')
            st.plotly_chart(fig_hours)
            
            # Di bagian analisis ketidakhadiran, ubah kode berikut:
            st.subheader("Analisis Ketidakhadiran")

            # Get all unique dates in attendance records
            working_dates = pd.date_range(start=df_attendance['date'].min(),
                                        end=df_attendance['date'].max(),
                                        freq='B')  # 'B' untuk business days (Senin-Jumat)

            # Create a DataFrame with all employee-date combinations
            employee_names = [emp['name'] for emp in employees]
            date_employee_combinations = pd.MultiIndex.from_product([working_dates, employee_names],
                                                                names=['date', 'employee_name'])
            full_attendance = pd.DataFrame(index=date_employee_combinations).reset_index()

            # Convert date column to datetime if it isn't already
            if not pd.api.types.is_datetime64_any_dtype(df_attendance['date']):
                df_attendance['date'] = pd.to_datetime(df_attendance['date'])

            # Merge with actual attendance and mark present/absent
            merged_attendance = pd.merge(full_attendance,
                                    df_attendance[['date', 'employee_name']],
                                    how='left',
                                    on=['date', 'employee_name'],
                                    indicator=True)

            # Mark attendance status
            merged_attendance['present'] = merged_attendance['_merge'] == 'both'

            # Calculate absences per employee
            absences = merged_attendance.groupby('employee_name')['present'].apply(
                lambda x: (~x).sum()
            ).reset_index()
            absences.columns = ['employee_name', 'absence_count']

            # Create bar chart for absences
            fig_absences = px.bar(absences,
                                x='employee_name',
                                y='absence_count',
                                labels={'employee_name': 'Nama Karyawan',
                                        'absence_count': 'Jumlah Ketidakhadiran'},
                                title='Jumlah Ketidakhadiran per Karyawan (Hari Kerja)')

            # Customize the chart
            fig_absences.update_traces(marker_color='#FF6B6B')  # Warna merah untuk ketidakhadiran
            fig_absences.update_layout(
                xaxis_title="Nama Karyawan",
                yaxis_title="Jumlah Ketidakhadiran",
                bargap=0.2,
                plot_bgcolor='white'
            )

            # Add gridlines
            fig_absences.update_yaxes(gridcolor='lightgray', gridwidth=0.5)

            st.plotly_chart(fig_absences)

            # Tambahan informasi statistik
            total_working_days = len(working_dates)
            avg_attendance_rate = 100 * (1 - absences['absence_count'].mean() / total_working_days)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Hari Kerja", f"{total_working_days} hari")
            with col2:
                st.metric("Rata-rata Kehadiran", f"{avg_attendance_rate:.1f}%")
            with col3:
                st.metric("Total Karyawan", len(employees))
            
        else:
            st.warning("Belum ada data presensi untuk dianalisis.")
    else:
        st.error("Gagal mengambil data karyawan atau presensi.")
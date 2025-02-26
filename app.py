from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import os
import cv2
from collections import Counter
from PIL import Image
import time  # Modul untuk menghitung waktu

# Konfigurasi Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load model YOLOv8
model = YOLO('models/best.pt')

def categorize_activity(detections):
    activities = {
        'Siswa - Siswa Sedang Duduk': ['Siswa', 'Berdiri', 'Laki2', 'Wanita', 'Bangku'],
        'Siswa Bersama Guru': ['Siswa', 'Buku', 'Meja', 'Bangku','Guru','Laki2','Wanita'],
        'Siswa - Siswa Berdoa': ['Duduk', 'Siswa', 'Laki2',"Wanita"],
        'Siswa - Siswa Sedang Duduk': ['Siswa', 'Duduk', 'Laki2', 'Wanita'],
        'Kegiatan Lainnya': ['Alat Tulis', 'Duduk']
    }

    detected_labels = set(d['label'] for d in detections)
    activity = 'Tidak dapat menentukan kegiatan'
    
    # Cek jika ada label "Guru"
    if 'Guru' in detected_labels:
        # Jika ada label "Guru", kategorikan sebagai "Belajar Bersama Guru"
        for category, labels in activities.items():
            if set(labels).intersection(detected_labels):
                activity = 'Siswa Bersama Guru'
                break
    else:
        # Jika tidak ada label "Guru", cek untuk "Belajar Mandiri"
        if any(label in detected_labels for label in ['Buku', 'Siswa', 'Tas', 'Bangku']):
            activity = 'Belajar Mandiri'
        else:
            # Jika tidak termasuk kategori khusus, cek kategori lainnya
            for category, labels in activities.items():
                if set(labels).intersection(detected_labels):
                    activity = category
                    break
    
    return activity


# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk upload gambar
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'images' not in request.files:
        return redirect(request.url)
    
    files = request.files.getlist('images')
    
    results = []
    all_detections = []
    all_labels = []
    
    # Buat variabel untuk menyimpan jumlah objek berdasarkan kategori
    count_gender = {'Laki-laki': 0, 'Perempuan': 0}
    count_role = {'Siswa': 0, 'Guru': 0}
    count_tools = {'Alat Tulis': 0, 'Buku': 0, 'Tas': 0}
    count_posisi = {'Berdiri': 0, 'Duduk': 0}

    # Buat dictionary untuk menyimpan pengelompokan berdasarkan kategori
    categorized_results = {
        'Belajar Bersama Guru': [],
        'Belajar Mandiri': [],
        'Olahraga': [],
        'Sosial': [],
        'Kegiatan Lainnya': [],
        'Siswa Sedang Berdiri': [],
        'Siswa Didekat Buku': [],
        'Siswa - Siswa Berdoa': [],
        'Siswa Sedang Duduk': [],
        'Siswa - Siswa Sedang Berdiri': [],
        'Siswa Bersama Guru': [],
        'Siswa - Siswa Sedang Duduk': [],
        'Kegiatan Lainnya': [],
        'Tidak dapat menentukan kegiatan': []
    }

    for file in files:
        if file.filename == '':
            continue
        
        # Simpan gambar yang diunggah ke folder static/uploads
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Load gambar dan jalankan YOLOv8 untuk deteksi objek
        img = cv2.imread(filepath)
        detections = []
        labels = []
        
        # Hitung waktu mulai deteksi
        start_time = time.time()

        result = model(img)  # Jalankan YOLOv8

        # Hitung waktu selesai deteksi
        end_time = time.time()
        detection_time = end_time - start_time
        
        # Dapatkan hasil deteksi objek
        for detection_result in result:
            for box in detection_result.boxes:
                label = detection_result.names[int(box.cls)]
                detection = {
                    'label': label,  # Nama objek
                    'confidence': float(box.conf),  # Confidence score
                    'bbox': [int(coord) for coord in box.xyxy[0]]  # Bounding box
                }
                detections.append(detection)
                labels.append(label)  # Tambahkan label ke list
        
        # Hitung jumlah kemunculan setiap label menggunakan Counter
        label_counts = Counter(labels)
        
        # Tambahkan jumlah berdasarkan kategori
        count_gender['Laki-laki'] += label_counts.get('Laki2', 0)
        count_gender['Perempuan'] += label_counts.get('Wanita', 0)
        count_role['Siswa'] += label_counts.get('Siswa', 0)
        count_role['Guru'] += label_counts.get('Guru', 0)
        count_tools['Alat Tulis'] += label_counts.get('Alat Tulis', 0)
        count_tools['Buku'] += label_counts.get('Buku', 0)
        count_tools['Tas'] += label_counts.get('Tas', 0)
        count_posisi['Berdiri'] += label_counts.get('Berdiri', 0)
        count_posisi['Duduk'] += label_counts.get('Duduk', 0)

        # Simpan gambar hasil deteksi
        detected_img = result[0].plot()
        detected_img_pil = Image.fromarray(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB))
        output_path = 'detected_' + file.filename
        detected_img_pil.save(os.path.join(app.config['UPLOAD_FOLDER'], output_path))
        
        # Tentukan kategori kegiatan
        activity = categorize_activity(detections)

        # Tambahkan hasil deteksi ke pengelompokan berdasarkan kategori
        categorized_results[activity].append({
            'filename': file.filename,
            'detected_image_path': output_path,
            'detections': detections,
            'total_detections': len(detections),
            'unique_objects': len(set(label for label in labels)),
            'detection_time': round(detection_time, 2)  # Tambahkan waktu deteksi
        })

        results.append({
            'filename': file.filename,
            'detected_image_path': output_path,
            'detections': detections,
            'total_detections': len(detections),
            'unique_objects': len(set(label for label in labels)),
            'activity': activity,
            'detection_time': round(detection_time, 2)  # Tambahkan waktu deteksi
        })
        all_detections.extend(detections)
        all_labels.extend(labels)
    
    # Hitung jumlah label untuk semua gambar
    label_counts = Counter(all_labels)
    total_count = sum(label_counts.values())
    
    # Kirim hasil deteksi, jalur gambar, label count, dan kategori kegiatan ke template HTML
    stats = [
        {
            'filename': result['filename'],
            'total_detections': result['total_detections'],
            'unique_objects': result['unique_objects'],
            'activity': result['activity'],
            'detection_time': result['detection_time']
        }
        for result in results
    ]
    
    return render_template('index.html', 
                           results=results, 
                           label_counts=label_counts, 
                           stats=stats, 
                           total_count=total_count, 
                           count_gender=count_gender, 
                           count_role=count_role, 
                           count_tools=count_tools,
                           count_posisi=count_posisi,
                           categorized_results=categorized_results)  # Tambahkan hasil pengelompokan

# Menjalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)

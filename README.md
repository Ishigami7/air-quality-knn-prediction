# 🌬️ Model KNN untuk Prediksi Kualitas Udara

Proyek machine learning menggunakan algoritma K-Nearest Neighbors (KNN) untuk memprediksi kualitas udara berdasarkan 6 parameter polutan utama.

## 📋 Deskripsi

Model ini dapat mengklasifikasikan kualitas udara ke dalam 3 kategori:
- **🟢 Baik**: Kualitas udara sehat untuk semua kelompak
- **🟡 Sedang**: Kualitas udara dapat diterima untuk sebagian besar orang
- **🔴 Buruk**: Kualitas udara tidak sehat

### Parameter Input:
- **PM10**: Particulate Matter 10 mikrometers (µg/m³)
- **PM2.5**: Particulate Matter 2.5 mikrometers (µg/m³)
- **SO2**: Sulfur Dioxide (µg/m³)
- **CO**: Carbon Monoxide (mg/m³)
- **O3**: Ozone (µg/m³)
- **NO2**: Nitrogen Dioxide (µg/m³)

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Ishigami7/air-quality-knn-prediction.git
cd air-quality-knn-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan Jupyter Notebook
```bash
jupyter notebook air_quality_knn_model.ipynb
```

## 📁 Struktur Proyek

```
air-quality-knn-prediction/
├── air_quality_knn_model.ipynb    # Notebook utama
├── sample_air_quality_data.csv    # Dataset contoh
├── utils.py                       # Utility functions
├── requirements.txt               # Dependencies
├── README.md                      # Dokumentasi
├── models/                        # Folder untuk menyimpan model
│   ├── knn_model.pkl             # Model terlatih
│   ├── scaler.pkl                # Feature scaler
│   └── encoder.pkl               # Label encoder
└── data/                         # Folder untuk dataset
    └── your_data.csv             # Dataset Anda
```

## 💡 Cara Menggunakan

### 1. Menyiapkan Dataset
Dataset harus berformat CSV dengan kolom:
```csv
PM10,PM2.5,SO2,CO,O3,NO2,Quality
45.2,25.1,15.3,8.2,85.4,32.1,Sedang
25.3,12.5,8.1,4.2,45.3,18.2,Baik
120.5,75.2,45.1,25.3,180.2,85.4,Buruk
```

### 2. Training Model
Jalankan semua cell di notebook `air_quality_knn_model.ipynb` secara berurutan.

### 3. Prediksi Data Baru
```python
import joblib
import numpy as np

# Load model components
model = joblib.load('models/knn_model.pkl')
scaler = joblib.load('models/scaler.pkl')
encoder = joblib.load('models/encoder.pkl')

# Contoh prediksi
input_data = np.array([[30, 15, 10, 5, 50, 20]])  # PM10, PM2.5, SO2, CO, O3, NO2
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)
result = encoder.inverse_transform(prediction)[0]

print(f"Prediksi kualitas udara: {result}")
```

## 🔧 Fitur Utama

### ✅ Data Preprocessing
- Handling missing values
- Feature scaling dengan StandardScaler
- Label encoding untuk target variable

### ✅ Model Training
- Cross-validation untuk mencari K optimal
- Hyperparameter tuning
- Model evaluation dengan confusion matrix

### ✅ Visualisasi
- Correlation heatmap antar parameter
- Distribution plots untuk setiap kategori
- Confusion matrix dan classification report

### ✅ Model Persistence
- Menyimpan model, scaler, dan encoder
- Load model untuk prediksi data baru

## 📊 Evaluasi Model

Model dievaluasi menggunakan:
- **Accuracy Score**: Tingkat akurasi keseluruhan
- **Confusion Matrix**: Detail prediksi per kategori
- **Classification Report**: Precision, Recall, F1-score
- **Cross-validation**: Validasi dengan K-fold CV

## 🛠️ Troubleshooting

### Error: "File not found"
- Pastikan file dataset berada di folder yang benar
- Periksa nama file dan ekstensi (.csv)

### Error: "Invalid column names"
- Pastikan header kolom sesuai: PM10, PM2.5, SO2, CO, O3, NO2, Quality
- Tidak ada spasi ekstra atau karakter khusus

### Error: "Data type error"
- Pastikan semua parameter numerik (PM10, PM2.5, dll.) berupa angka
- Ganti koma desimal dengan titik jika perlu

### Model akurasi rendah
- Periksa kualitas dataset (apakah ada outlier?)
- Coba tambah lebih banyak data training
- Eksperimen dengan nilai K yang berbeda

## 📈 Tips Optimalisasi

1. **Dataset Quality**:
   - Gunakan minimal 100+ sampel untuk setiap kategori
   - Pastikan distribusi data seimbang
   - Hilangkan outlier yang ekstrem

2. **Feature Engineering**:
   - Pertimbangkan membuat fitur baru (rasio PM2.5/PM10)
   - Normalisasi berdasarkan standar kualitas udara lokal

3. **Model Tuning**:
   - Coba algoritma lain (Random Forest, SVM)
   - Eksperimen dengan different distance metrics
   - Gunakan weighted KNN untuk data tidak seimbang

## 🤝 Contributing

1. Fork repository
2. Buat branch feature (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Contact

Your Name - [@Ishigami7](https://github.com/Ishigami7)

Project Link: [https://github.com/Ishigami7/air-quality-knn-prediction](https://github.com/Ishigami7/air-quality-knn-prediction)

## 🙏 Acknowledgments

- [scikit-learn](https://scikit-learn.org/) - Machine learning library
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [matplotlib](https://matplotlib.org/) - Data visualization
- WHO Air Quality Guidelines untuk referensi standar kualitas udara
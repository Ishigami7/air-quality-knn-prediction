"""
Utility functions untuk model KNN Prediksi Kualitas Udara
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def load_and_validate_data(file_path):
    """
    Load dan validasi dataset kualitas udara
    
    Parameters:
    file_path (str): path ke file CSV
    
    Returns:
    pd.DataFrame: validated dataframe
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dataset berhasil dimuat: {df.shape}")
        
        # Validasi kolom yang diperlukan
        required_columns = ['PM10', 'PM2.5', 'SO2', 'CO', 'O3', 'NO2', 'Quality']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"❌ Kolom tidak ditemukan: {missing_columns}")
        
        # Validasi tipe data
        numeric_columns = ['PM10', 'PM2.5', 'SO2', 'CO', 'O3', 'NO2']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Cek missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"⚠️  Ditemukan {missing_count} missing values")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ File tidak ditemukan: {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

def preprocess_data(df):
    """
    Preprocessing data kualitas udara
    
    Parameters:
    df (pd.DataFrame): raw dataframe
    
    Returns:
    pd.DataFrame: preprocessed dataframe
    """
    df_processed = df.copy()
    
    # Handle missing values dengan median
    numeric_columns = ['PM10', 'PM2.5', 'SO2', 'CO', 'O3', 'NO2']
    for col in numeric_columns:
        if df_processed[col].isnull().sum() > 0:
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
            print(f"🔧 {col}: filled dengan median = {median_val:.2f}")
    
    # Standardisasi kategori target
    category_mapping = {
        'good': 'Baik', 'Good': 'Baik', 'GOOD': 'Baik',
        'moderate': 'Sedang', 'Moderate': 'Sedang', 'MODERATE': 'Sedang',
        'bad': 'Buruk', 'Bad': 'Buruk', 'BAD': 'Buruk',
        'poor': 'Buruk', 'Poor': 'Buruk', 'POOR': 'Buruk'
    }
    df_processed['Quality'] = df_processed['Quality'].replace(category_mapping)
    
    # Validasi rentang nilai (opsional - sesuaikan dengan domain knowledge)
    for col in numeric_columns:
        if (df_processed[col] < 0).any():
            print(f"⚠️  Ditemukan nilai negatif di kolom {col}")
            df_processed[col] = df_processed[col].abs()
    
    return df_processed

def plot_model_performance(y_true, y_pred, target_names):
    """
    Plot confusion matrix dan classification report
    
    Parameters:
    y_true: true labels
    y_pred: predicted labels  
    target_names: nama kategori target
    """
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Model KNN')
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.tight_layout()
    plt.show()
    
    # Classification Report
    print("📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

def plot_feature_importance_correlation(df):
    """
    Plot korelasi antar fitur kualitas udara
    
    Parameters:
    df (pd.DataFrame): dataframe dengan fitur numerik
    """
    numeric_features = ['PM10', 'PM2.5', 'SO2', 'CO', 'O3', 'NO2']
    correlation_matrix = df[numeric_features].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Korelasi Antar Parameter Kualitas Udara')
    plt.tight_layout()
    plt.show()

def save_model_components(model, scaler, encoder, model_path='models/'):
    """
    Simpan semua komponen model
    
    Parameters:
    model: trained model
    scaler: fitted scaler
    encoder: fitted label encoder
    model_path: path untuk menyimpan model
    """
    import os
    
    # Buat direktori jika belum ada
    os.makedirs(model_path, exist_ok=True)
    
    # Simpan komponen
    joblib.dump(model, f'{model_path}knn_model.pkl')
    joblib.dump(scaler, f'{model_path}scaler.pkl') 
    joblib.dump(encoder, f'{model_path}encoder.pkl')
    
    print(f"💾 Model components disimpan di: {model_path}")

def load_model_components(model_path='models/'):
    """
    Load semua komponen model
    
    Parameters:
    model_path: path model
    
    Returns:
    tuple: (model, scaler, encoder)
    """
    try:
        model = joblib.load(f'{model_path}knn_model.pkl')
        scaler = joblib.load(f'{model_path}scaler.pkl')
        encoder = joblib.load(f'{model_path}encoder.pkl')
        
        print("✅ Model components berhasil dimuat")
        return model, scaler, encoder
        
    except FileNotFoundError as e:
        print(f"❌ File model tidak ditemukan: {e}")
        return None, None, None

def predict_single_sample(model, scaler, encoder, pm10, pm25, so2, co, o3, no2):
    """
    Prediksi kualitas udara untuk satu sampel
    
    Parameters:
    model: trained model
    scaler: fitted scaler
    encoder: fitted encoder
    pm10, pm25, so2, co, o3, no2: parameter input
    
    Returns:
    tuple: (predicted_class, probabilities)
    """
    # Buat input array
    input_data = np.array([[pm10, pm25, so2, co, o3, no2]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Prediksi
    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)
    
    # Convert ke label
    predicted_class = encoder.inverse_transform(prediction)[0]
    
    return predicted_class, probabilities[0]

def generate_sample_data(n_samples=100, save_path='sample_data.csv'):
    """
    Generate sample dataset untuk testing
    
    Parameters:
    n_samples: jumlah sampel yang akan dibuat
    save_path: path untuk menyimpan file
    """
    np.random.seed(42)
    
    data = []
    
    for _ in range(n_samples):
        # Random pilih kategori
        category = np.random.choice(['Baik', 'Sedang', 'Buruk'])
        
        if category == 'Baik':
            # Nilai polusi rendah
            pm10 = np.random.normal(25, 8)
            pm25 = np.random.normal(12, 4)
            so2 = np.random.normal(8, 3)
            co = np.random.normal(4, 2)
            o3 = np.random.normal(45, 10)
            no2 = np.random.normal(18, 5)
        elif category == 'Sedang':
            # Nilai polusi sedang
            pm10 = np.random.normal(65, 15)
            pm25 = np.random.normal(35, 8)
            so2 = np.random.normal(25, 8)
            co = np.random.normal(12, 4)
            o3 = np.random.normal(95, 20)
            no2 = np.random.normal(45, 10)
        else:  # Buruk
            # Nilai polusi tinggi
            pm10 = np.random.normal(110, 20)
            pm25 = np.random.normal(65, 12)
            so2 = np.random.normal(45, 12)
            co = np.random.normal(22, 6)
            o3 = np.random.normal(165, 25)
            no2 = np.random.normal(75, 15)
        
        # Pastikan nilai tidak negatif
        pm10 = max(0, pm10)
        pm25 = max(0, pm25)
        so2 = max(0, so2)
        co = max(0, co)
        o3 = max(0, o3)
        no2 = max(0, no2)
        
        data.append([pm10, pm25, so2, co, o3, no2, category])
    
    # Buat DataFrame
    df = pd.DataFrame(data, columns=['PM10', 'PM2.5', 'SO2', 'CO', 'O3', 'NO2', 'Quality'])
    
    # Simpan ke CSV
    df.to_csv(save_path, index=False)
    print(f"📁 Sample data berhasil dibuat: {save_path}")
    print(f"   Total sampel: {len(df)}")
    print(f"   Distribusi: {df['Quality'].value_counts().to_dict()}")
    
    return df
# Sistem Klasifikasi Spam Email Menggunakan Random Forest

Sistem ini menggunakan algoritma **Random Forest** untuk mengklasifikasikan email sebagai Spam atau Ham (email normal).

## 📋 Struktur Project

```
AI PAPER/
├── data/
│   └── spam.csv                 # Dataset email spam
├── src/
│   ├── main.py                 # Script utama
│   ├── data_preprocessing.py   # Module preprocessing data
│   ├── model.py                # Module model training
│   └── evaluation.py           # Module evaluasi model
├── models/                      # Folder untuk menyimpan model
├── results/                     # Folder untuk hasil visualisasi
└── requirements.txt             # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
```bash
cd src
python main.py
```

### 3. Output
Program akan menghasilkan:
- Model yang sudah dilatih (`models/spam_classifier_rf.pkl`)
- Visualisasi hasil:
  - `results/confusion_matrix.png`
  - `results/roc_curve.png`
  - `results/precision_recall_curve.png`
  - `results/feature_importance.png`

## 📊 Fitur Utama

### 1. Data Preprocessing
- Pembersihan teks (hapus URL, email, karakter khusus)
- Normalisasi teks (lowercase, hapus whitespace)
- Ekstraksi fitur menggunakan TF-IDF
- Split data training-testing (80-20)

### 2. Model Training
- Baseline model dengan n_estimators=100
- Hyperparameter tuning menggunakan GridSearchCV
- Optimal parameters:
  - `n_estimators`: jumlah pohon keputusan
  - `max_depth`: kedalaman maksimal pohon
  - `min_samples_split`: minimal sampel untuk split
  - `min_samples_leaf`: minimal sampel di leaf

### 3. Evaluasi Model
- **Accuracy**: Akurasi keseluruhan
- **Precision**: Akurasi prediksi spam yang benar
- **Recall**: Kemampuan mendeteksi email spam
- **F1-Score**: Harmonic mean precision dan recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Matrix kesalahan prediksi

### 4. Feature Analysis
- Identifikasi top 20 fitur paling penting
- Menampilkan kata-kata yang paling berpengaruh dalam klasifikasi

## 📈 Dataset

Dataset `spam.csv` berisi:
- **Kolom 1 (v1)**: Label (ham atau spam)
- **Kolom 2 (v2)**: Teks email/SMS

Format:
```
label,text
ham,Go until jurong point, crazy..
spam,Free entry in 2 a wkly comp to win FA Cup...
```

## 🔧 Module Description

### `data_preprocessing.py`
- `load_data()`: Membaca file CSV
- `clean_text()`: Membersihkan teks
- `preprocess_data()`: Preprocessing DataFrame
- `create_features()`: Membuat fitur TF-IDF
- `prepare_data()`: Pipeline lengkap preprocessing

### `model.py`
- `create_random_forest()`: Membuat model
- `train_model()`: Melatih model
- `hyperparameter_tuning()`: Tuning hyperparameter
- `save_model()`: Menyimpan model
- `load_model()`: Memuat model
- `get_feature_importance()`: Analisis fitur penting

### `evaluation.py`
- `get_predictions()`: Mendapatkan prediksi
- `evaluate_model()`: Menghitung metrik evaluasi
- `print_evaluation_report()`: Cetak laporan lengkap
- `plot_confusion_matrix()`: Plot confusion matrix
- `plot_roc_curve()`: Plot ROC curve
- `plot_precision_recall_curve()`: Plot PR curve
- `plot_feature_importance()`: Plot feature importance

## 💡 Penggunaan Advanced

### Memprediksi Email Baru

```python
from src.main import predict_email

result = predict_email("Free entry to win! Click here!")
print(result['prediction'])  # Output: SPAM
print(result['confidence'])  # Output: 0.95
```

### Custom Training

```python
from src.data_preprocessing import prepare_data
from src.model import train_model, save_model

# Prepare data
data = prepare_data('data/spam.csv', test_size=0.2)

# Train with custom parameters
model = train_model(data['X_train'], data['y_train'], n_estimators=200)

# Save
save_model(model, 'models/custom_model.pkl')
```

## 🎯 Expected Performance

Typical performance metrics:
- **Accuracy**: 95-97%
- **Precision**: 93-95%
- **Recall**: 91-93%
- **F1-Score**: 92-94%
- **ROC-AUC**: 0.98-0.99

## 📝 Catatan Penting

1. **Dataset Balance**: Pastikan dataset seimbang antara spam dan ham
2. **Feature Engineering**: TF-IDF optimal untuk text classification
3. **Hyperparameter**: Sesuaikan berdasarkan hardware dan dataset size
4. **Cross Validation**: GridSearchCV menggunakan 5-fold CV
5. **Random State**: Gunakan untuk reproducibility hasil

## 🐛 Troubleshooting

### Memory Error
- Kurangi `max_features` di TfidfVectorizer
- Gunakan `n_jobs=1` untuk mengurangi parallel processing

### Slow Training
- Kurangi jumlah parameter di GridSearchCV
- Gunakan `n_jobs=-1` untuk parallel processing

### Model Overfitting
- Tambah `min_samples_split` dan `min_samples_leaf`
- Kurangi `max_depth`
- Tambah `max_df` di TfidfVectorizer

## 👨‍💻 Author
AI Paper Project

## 📄 License
MIT License

---

**Catatan**: Untuk hasil terbaik, pastikan dataset cukup besar (minimal 5000 sampel) dan seimbang antara kelas.

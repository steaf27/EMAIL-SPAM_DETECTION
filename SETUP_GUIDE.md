# 🚀 Setup Guide - Sistem Klasifikasi Spam Email

## Persyaratan Sistem

- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- 2GB RAM minimum
- Storage 500MB

## Step-by-Step Installation

### 1. Clone/Download Project

```bash
# Navigasi ke folder project
cd "AI PAPER"
```

### 2. Buat Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Verifikasi instalasi:
```bash
pip list
```

### 4. Persiapan Data

Pastikan file `data/spam.csv` ada dan memiliki format:
```
label,text
ham,message text...
spam,message text...
```

## ⚙️ Konfigurasi

Edit file `src/config.py` untuk mengubah parameter:

```python
# Contoh konfigurasi
TEST_SIZE = 0.2              # 20% untuk testing
MAX_FEATURES = 5000          # Maksimal 5000 fitur
NGRAM_RANGE = (1, 2)         # Gunakan unigram dan bigram
```

## 🏃 Menjalankan Project

### Training Model

```bash
cd src
python main.py
```

Output yang dihasilkan:
- ✅ Model terlatih → `models/spam_classifier_rf.pkl`
- ✅ Vectorizer → `models/vectorizer.pkl`
- ✅ Visualisasi → `results/` folder
- ✅ Laporan performa

### Prediksi Email Baru

Dari folder project root:

```bash
python predict.py
```

Fitur:
- Prediksi dengan 8 contoh email
- Mode interaktif untuk input email custom
- Output confidence dan probabilitas

## 📊 Struktur File Output

Setelah training, folder akan terstruktur:

```
AI PAPER/
├── data/
│   └── spam.csv
├── src/
│   ├── main.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── evaluation.py
│   └── config.py
├── models/                    # Folder baru
│   ├── spam_classifier_rf.pkl
│   └── vectorizer.pkl
├── results/                   # Folder baru
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── feature_importance.png
├── README.md
├── requirements.txt
├── predict.py
└── .gitignore
```

## 🔍 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'sklearn'"
**Solution:**
```bash
pip install scikit-learn
```

### Problem: "FileNotFoundError: data/spam.csv"
**Solution:**
- Pastikan file `spam.csv` berada di folder `data/`
- Periksa path relatif dalam script

### Problem: Memory Error
**Solution:**
- Edit `src/config.py`:
  ```python
  MAX_FEATURES = 3000  # Kurangi dari 5000
  MIN_DF = 5           # Naikkan dari 2
  ```
- Atau jalankan di mesin dengan RAM lebih besar

### Problem: Training terlalu lambat
**Solution:**
- Edit `src/config.py`:
  ```python
  CV_FOLDS = 3         # Kurangi dari 5
  PARAM_GRID['n_estimators'] = [100]  # Satu nilai saja
  ```

## 💡 Tips Optimization

### Untuk Akurasi Lebih Tinggi:
1. Gunakan dataset yang lebih besar
2. Naikkan `MAX_FEATURES` menjadi 10000
3. Tambah CV_FOLDS menjadi 10

### Untuk Kecepatan Lebih Cepat:
1. Kurangi `MAX_FEATURES` menjadi 2000
2. Kurangi `CV_FOLDS` menjadi 3
3. Edit `PARAM_GRID` untuk tuning yang lebih sederhana

## 📚 Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

## ✅ Verification Checklist

- [ ] Python 3.8+ terinstall
- [ ] Virtual environment aktif
- [ ] Dependencies terinstall (`pip list` untuk verifikasi)
- [ ] File `data/spam.csv` ada
- [ ] Bisa menjalankan `python main.py` tanpa error
- [ ] Folder `models/` dan `results/` terbuat
- [ ] Bisa menjalankan `python predict.py`

## 🆘 Bantuan

Jika mengalami masalah:

1. Cek error message dengan seksama
2. Pastikan semua dependencies terinstall
3. Coba jalankan di Python environment yang baru
4. Cek versi Python: `python --version`

---

**Happy Spam Filtering! 🎯**

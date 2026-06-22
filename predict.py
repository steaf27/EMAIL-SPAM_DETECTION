"""
Script contoh untuk menggunakan model yang sudah dilatih
untuk memprediksi email baru
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import clean_text
from model import load_model
import joblib


def predict_email(email_text, vectorizer, model):
    """
    Memprediksi apakah email adalah spam atau bukan

    Parameters:
    -----------
    email_text : str
        Teks email yang akan diprediksi
    vectorizer : TfidfVectorizer
        Vectorizer yang sudah dilatih
    model : RandomForestClassifier
        Model yang sudah dilatih

    Returns:
    --------
    dict
        Hasil prediksi
    """
    # Clean text
    cleaned_text = clean_text(email_text)

    # Vectorize
    X = vectorizer.transform([cleaned_text])

    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    return {
        "email": email_text,
        "cleaned_text": cleaned_text,
        "prediction": "SPAM 🚨" if prediction == 1 else "HAM ✓",
        "confidence": max(probability) * 100,
        "probability_ham": probability[0] * 100,
        "probability_spam": probability[1] * 100,
    }


def main():
    """
    Fungsi utama untuk contoh prediksi
    """

    print("\n" + "=" * 80)
    print("CONTOH PREDIKSI SPAM EMAIL")
    print("=" * 80 + "\n")

    # Load model
    model_path = os.path.join(
        os.path.dirname(__file__), "models", "spam_classifier_rf.pkl"
    )

    vectorizer_path = os.path.join(
        os.path.dirname(__file__), "models", "vectorizer.pkl"
    )

    if not os.path.exists(model_path):
        print("❌ Model tidak ditemukan!")
        print(f"   Path: {model_path}")
        print("   Jalankan main.py terlebih dahulu untuk melatih model.")
        return

    print("📦 Memuat model...")
    model = load_model(model_path)

    # Load vectorizer
    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
        print("📦 Memuat vectorizer...")
    else:
        print("⚠️  Vectorizer tidak ditemukan, jalankan training terlebih dahulu")
        return

    # Contoh email
    test_emails = [
        "Congratulations! You've won a FREE iPhone. Click here to claim your prize now!",
        "Hi John, Can we schedule a meeting tomorrow at 2 PM? Thanks!",
        "URGENT! Your account has been compromised. Click here to verify your identity.",
        "Remember the coffee meeting tomorrow at 3 PM. See you then!",
        "Get cheap medications now! No prescription needed. Order today!",
        "The quarterly report is ready for review. Please check the attached file.",
        "CLICK HERE for instant loans! Bad credit OK! Get cash now!",
        "Hi, just checking in. How have you been?",
    ]

    print("\n" + "=" * 80)
    print("HASIL PREDIKSI")
    print("=" * 80 + "\n")

    for i, email in enumerate(test_emails, 1):
        result = predict_email(email, vectorizer, model)

        print(f"[Email {i}]")
        print(f"Teks: {result['email'][:70]}...")
        print(f"Prediksi: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"  - Probability HAM : {result['probability_ham']:.2f}%")
        print(f"  - Probability SPAM: {result['probability_spam']:.2f}%")
        print()

    print("=" * 80 + "\n")

    # Interactive prediction
    print("\n🎯 PREDIKSI INTERAKTIF")
    print("-" * 80)

    while True:
        print("\nMasukkan teks email (atau 'quit' untuk keluar):")
        user_input = input("> ").strip()

        if user_input.lower() == "quit":
            print("Terima kasih! Program selesai.")
            break

        if not user_input:
            print("❌ Masukan tidak boleh kosong!")
            continue

        result = predict_email(user_input, vectorizer, model)

        print(f"\n📊 HASIL ANALISIS:")
        print(f"   Prediksi: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.2f}%")
        print(f"   Probability HAM : {result['probability_ham']:.2f}%")
        print(f"   Probability SPAM: {result['probability_spam']:.2f}%")


if __name__ == "__main__":
    main()

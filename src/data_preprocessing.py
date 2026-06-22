"""
Module untuk preprocessing data email spam
"""

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """
    Membaca file CSV dan mengembalikan DataFrame

    Parameters:
    -----------
    filepath : str
        Path ke file spam.csv

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame dengan kolom 'label' dan 'text'
    """
    try:
        df = pd.read_csv(filepath, usecols=[0, 1], encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, usecols=[0, 1], encoding="latin-1")

    df.columns = ["label", "text"]
    return df


def clean_text(text):
    """
    Membersihkan teks dari karakter khusus, URL, dan karakter tidak perlu

    Parameters:
    -----------
    text : str
        Teks yang akan dibersihkan

    Returns:
    --------
    str
        Teks yang sudah dibersihkan
    """
    # Hapus URL
    text = re.sub(r"http\S+|www\S+", "", text)

    # Hapus email
    text = re.sub(r"\S+@\S+", "", text)

    # Hapus karakter khusus dan angka
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Konversi ke lowercase
    text = text.lower()

    # Hapus whitespace berlebih
    text = " ".join(text.split())

    return text


def preprocess_data(df):
    """
    Melakukan preprocessing pada DataFrame

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame dengan kolom 'label' dan 'text'

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame yang sudah diproses
    """
    # Hapus baris dengan nilai kosong
    df = df.dropna()

    # Reset index
    df = df.reset_index(drop=True)

    # Bersihkan teks
    df["text"] = df["text"].apply(clean_text)

    # Hapus baris dengan teks kosong
    df = df[df["text"].str.len() > 0]
    df = df.reset_index(drop=True)

    return df


def create_features(X_train, X_test):
    """
    Membuat fitur TF-IDF dari teks

    Parameters:
    -----------
    X_train : pd.Series
        Data training
    X_test : pd.Series
        Data testing

    Returns:
    --------
    tuple
        (X_train_tfidf, X_test_tfidf, vectorizer)
    """
    # Inisialisasi TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Maksimal 5000 fitur
        min_df=2,  # Minimal dokumen yang mengandung term
        max_df=0.8,  # Maksimal proporsi dokumen
        ngram_range=(1, 2),  # Unigram dan bigram
    )

    # Fit dan transform
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, vectorizer


def prepare_data(filepath, test_size=0.2, random_state=42):
    """
    Mempersiapkan data untuk training model

    Parameters:
    -----------
    filepath : str
        Path ke file CSV
    test_size : float
        Proporsi data untuk testing
    random_state : int
        Random state untuk reproducibility

    Returns:
    --------
    dict
        Dictionary berisi X_train, X_test, y_train, y_test, dan vectorizer
    """
    # Load data
    df = load_data(filepath)

    # Preprocessing
    df = preprocess_data(df)

    # Encode label (ham=0, spam=1)
    df["label"] = (df["label"] == "spam").astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    # Create features
    X_train_tfidf, X_test_tfidf, vectorizer = create_features(X_train, X_test)

    return {
        "X_train": X_train_tfidf,
        "X_test": X_test_tfidf,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "vectorizer": vectorizer,
        "original_df": df,
    }

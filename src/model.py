"""
Module untuk membuat dan melatih model Random Forest
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os


def create_random_forest(n_estimators=100, random_state=42, **kwargs):
    """
    Membuat model Random Forest Classifier

    Parameters:
    -----------
    n_estimators : int
        Jumlah pohon dalam forest
    random_state : int
        Random state untuk reproducibility
    **kwargs
        Parameter tambahan untuk RandomForestClassifier

    Returns:
    --------
    model : RandomForestClassifier
        Model Random Forest yang sudah diinisialisasi
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,  # Gunakan semua core
        **kwargs,
    )
    return model


def train_model(X_train, y_train, n_estimators=100):
    """
    Melatih model Random Forest

    Parameters:
    -----------
    X_train : sparse matrix
        Data training (TF-IDF)
    y_train : array
        Label training
    n_estimators : int
        Jumlah pohon dalam forest

    Returns:
    --------
    model : RandomForestClassifier
        Model yang sudah dilatih
    """
    model = create_random_forest(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    return model


def hyperparameter_tuning(X_train, y_train, cv=5):
    """
    Melakukan hyperparameter tuning menggunakan GridSearchCV

    Parameters:
    -----------
    X_train : sparse matrix
        Data training (TF-IDF)
    y_train : array
        Label training
    cv : int
        Cross-validation folds

    Returns:
    --------
    best_model : RandomForestClassifier
        Model terbaik setelah tuning
    best_params : dict
        Parameter terbaik
    cv_results : dict
        Hasil cross-validation
    """
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [20, 30, 40, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(
        rf, param_grid, cv=cv, n_jobs=-1, verbose=1, scoring="f1_weighted"
    )

    grid_search.fit(X_train, y_train)

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.cv_results_,
    )


def save_model(model, filepath):
    """
    Menyimpan model ke file

    Parameters:
    -----------
    model : RandomForestClassifier
        Model yang akan disimpan
    filepath : str
        Path untuk menyimpan file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model berhasil disimpan ke: {filepath}")


def load_model(filepath):
    """
    Memuat model dari file

    Parameters:
    -----------
    filepath : str
        Path file model

    Returns:
    --------
    model : RandomForestClassifier
        Model yang dimuat
    """
    model = joblib.load(filepath)
    print(f"Model berhasil dimuat dari: {filepath}")
    return model


def get_feature_importance(model, vectorizer, top_n=20):
    """
    Mendapatkan feature importance dari model

    Parameters:
    -----------
    model : RandomForestClassifier
        Model yang sudah dilatih
    vectorizer : TfidfVectorizer
        Vectorizer yang digunakan
    top_n : int
        Jumlah top features yang akan ditampilkan

    Returns:
    --------
    features : list
        List dari fitur penting
    importances : array
        Nilai importance untuk setiap fitur
    """
    # Dapatkan feature names
    feature_names = vectorizer.get_feature_names_out()

    # Dapatkan importance
    importances = model.feature_importances_

    # Sort dan dapatkan top N
    indices = importances.argsort()[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    return top_features, top_importances

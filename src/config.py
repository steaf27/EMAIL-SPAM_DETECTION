"""
Konfigurasi parameter untuk sistem klasifikasi spam email
"""

# ========================
# DATA PREPROCESSING
# ========================
TEST_SIZE = 0.2  # Proporsi test set
RANDOM_STATE = 42  # Random seed untuk reproducibility

# TF-IDF Parameters
MAX_FEATURES = 5000  # Maksimal feature
MIN_DF = 2  # Minimal document frequency
MAX_DF = 0.8  # Maksimal document frequency
NGRAM_RANGE = (1, 2)  # Unigram dan bigram

# ========================
# MODEL TRAINING
# ========================
# Baseline Parameters
BASELINE_N_ESTIMATORS = 100

# GridSearchCV Parameters
PARAM_GRID = {
    "n_estimators": [50, 100, 150],
    "max_depth": [20, 30, 40, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

CV_FOLDS = 5  # Cross-validation folds
SCORING = "f1_weighted"  # Scoring metric

# ========================
# RANDOM FOREST PARAMETERS
# ========================
RF_N_JOBS = -1  # Gunakan semua cores
RF_VERBOSE = 1  # Verbose level
RF_RANDOM_STATE = 42

# ========================
# EVALUATION
# ========================
TOP_N_FEATURES = 20  # Jumlah top features yang ditampilkan

# ========================
# PATHS
# ========================
DATA_PATH = "../data/spam.csv"
MODEL_SAVE_PATH = "../models/spam_classifier_rf.pkl"
VECTORIZER_SAVE_PATH = "../models/vectorizer.pkl"
RESULTS_DIR = "../results"

# ========================
# VISUALIZATION
# ========================
FIGURE_DPI = 300
FIGURE_SIZE_DEFAULT = (10, 8)
FIGURE_SIZE_SQUARE = (8, 8)

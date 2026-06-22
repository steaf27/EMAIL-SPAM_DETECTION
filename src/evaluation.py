"""
Module untuk evaluasi model
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns


def get_predictions(model, X_test):
    """
    Mendapatkan prediksi dari model

    Parameters:
    -----------
    model : RandomForestClassifier
        Model yang sudah dilatih
    X_test : sparse matrix
        Data testing

    Returns:
    --------
    y_pred : array
        Prediksi class
    y_pred_proba : array
        Probabilitas prediksi
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return y_pred, y_pred_proba


def evaluate_model(y_test, y_pred, y_pred_proba):
    """
    Mengevaluasi performa model

    Parameters:
    -----------
    y_test : array
        Label testing (ground truth)
    y_pred : array
        Prediksi class
    y_pred_proba : array
        Probabilitas prediksi

    Returns:
    --------
    metrics : dict
        Dictionary berisi berbagai metrik evaluasi
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
    }

    return metrics


def print_evaluation_report(y_test, y_pred, y_pred_proba):
    """
    Mencetak laporan evaluasi lengkap

    Parameters:
    -----------
    y_test : array
        Label testing (ground truth)
    y_pred : array
        Prediksi class
    y_pred_proba : array
        Probabilitas prediksi
    """
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)

    print("\n" + "=" * 60)
    print("LAPORAN EVALUASI MODEL SPAM CLASSIFICATION")
    print("=" * 60)
    print(f"\nAccuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-Score  : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")

    print("\n" + "-" * 60)
    print("Confusion Matrix:")
    print("-" * 60)
    print(f"True Negatives  : {metrics['tn']}")
    print(f"False Positives : {metrics['fp']}")
    print(f"False Negatives : {metrics['fn']}")
    print(f"True Positives  : {metrics['tp']}")

    print("\n" + "-" * 60)
    print("Classification Report:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=["Ham (0)", "Spam (1)"]))
    print("=" * 60 + "\n")

    return metrics


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    """
    Membuat plot confusion matrix

    Parameters:
    -----------
    y_test : array
        Label testing (ground truth)
    y_pred : array
        Prediksi class
    save_path : str, optional
        Path untuk menyimpan gambar
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()


def plot_roc_curve(y_test, y_pred_proba, save_path=None):
    """
    Membuat plot ROC curve

    Parameters:
    -----------
    y_test : array
        Label testing (ground truth)
    y_pred_proba : array
        Probabilitas prediksi
    save_path : str, optional
        Path untuk menyimpan gambar
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"ROC curve saved to: {save_path}")

    plt.show()


def plot_precision_recall_curve(y_test, y_pred_proba, save_path=None):
    """
    Membuat plot Precision-Recall curve

    Parameters:
    -----------
    y_test : array
        Label testing (ground truth)
    y_pred_proba : array
        Probabilitas prediksi
    save_path : str, optional
        Path untuk menyimpan gambar
    """
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Precision-Recall curve saved to: {save_path}")

    plt.show()


def plot_feature_importance(top_features, top_importances, save_path=None):
    """
    Membuat plot feature importance

    Parameters:
    -----------
    top_features : list
        Daftar fitur penting
    top_importances : array
        Nilai importance untuk setiap fitur
    save_path : str, optional
        Path untuk menyimpan gambar
    """
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_importances, color="steelblue")
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel("Feature Importance")
    plt.title("Top 20 Most Important Features", fontweight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Feature importance plot saved to: {save_path}")

    plt.show()

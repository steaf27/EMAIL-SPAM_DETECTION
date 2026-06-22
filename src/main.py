"""
Script utama untuk melatih dan mengevaluasi model Random Forest
untuk klasifikasi Spam Email
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import modules
from data_preprocessing import prepare_data
from model import train_model, hyperparameter_tuning, save_model, get_feature_importance
from evaluation import get_predictions, print_evaluation_report, (
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,
    plot_feature_importance
)


def main():
    """
    Fungsi utama untuk menjalankan pipeline klasifikasi spam email
    """
    
    print("\n" + "="*70)
    print("SISTEM KLASIFIKASI SPAM EMAIL MENGGUNAKAN RANDOM FOREST")
    print("="*70)
    
    # ============================================================
    # 1. DATA PREPROCESSING
    # ============================================================
    print("\n[1] Melakukan preprocessing data...")
    print("-" * 70)
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'spam.csv')
    
    if not os.path.exists(data_path):
        print(f"❌ File tidak ditemukan: {data_path}")
        return
    
    # Prepare data
    data = prepare_data(data_path, test_size=0.2, random_state=42)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    vectorizer = data['vectorizer']
    original_df = data['original_df']
    
    print(f"✓ Data berhasil dipersiapkan")
    print(f"  - Total sampel: {len(original_df)}")
    print(f"  - Training set: {X_train.shape[0]} ({X_train.shape[0]/len(original_df)*100:.1f}%)")
    print(f"  - Testing set: {X_test.shape[0]} ({X_test.shape[0]/len(original_df)*100:.1f}%)")
    print(f"  - Feature dimensionality: {X_train.shape[1]}")
    print(f"  - Distribusi label (Ham:Spam): {sum(y_train==0)}:{sum(y_train==1)} (train)")
    
    # ============================================================
    # 2. MODEL TRAINING (Baseline)
    # ============================================================
    print("\n[2] Melatih model Random Forest (baseline)...")
    print("-" * 70)
    
    model = train_model(X_train, y_train, n_estimators=100)
    print("✓ Model baseline berhasil dilatih")
    
    # ============================================================
    # 3. HYPERPARAMETER TUNING
    # ============================================================
    print("\n[3] Melakukan hyperparameter tuning...")
    print("-" * 70)
    
    best_model, best_params, cv_results = hyperparameter_tuning(X_train, y_train, cv=5)
    
    print("✓ Tuning selesai")
    print(f"  - Best parameters:")
    for param, value in best_params.items():
        print(f"    • {param}: {value}")
    
    # ============================================================
    # 4. MODEL EVALUATION
    # ============================================================
    print("\n[4] Mengevaluasi model...")
    print("-" * 70)
    
    # Prediksi menggunakan model terbaik
    y_pred, y_pred_proba = get_predictions(best_model, X_test)
    
    # Print detailed evaluation report
    metrics = print_evaluation_report(y_test, y_pred, y_pred_proba)
    
    # ============================================================
    # 5. FEATURE IMPORTANCE
    # ============================================================
    print("\n[5] Menganalisis feature importance...")
    print("-" * 70)
    
    top_features, top_importances = get_feature_importance(best_model, vectorizer, top_n=20)
    
    print("✓ Top 20 fitur paling penting:")
    for i, (feature, importance) in enumerate(zip(top_features, top_importances), 1):
        print(f"  {i:2d}. {feature:20s} - {importance:.6f}")
    
    # ============================================================
    # 6. SAVE MODEL
    # ============================================================
    print("\n[6] Menyimpan model...")
    print("-" * 70)
    
    model_save_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'models', 
        'spam_classifier_rf.pkl'
    )
    
    save_model(best_model, model_save_path)
    
    # ============================================================
    # 7. VISUALISASI
    # ============================================================
    print("\n[7] Membuat visualisasi hasil...")
    print("-" * 70)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, save_path=cm_path)
    
    # Plot ROC curve
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plot_roc_curve(y_test, y_pred_proba, save_path=roc_path)
    
    # Plot Precision-Recall curve
    pr_path = os.path.join(output_dir, 'precision_recall_curve.png')
    plot_precision_recall_curve(y_test, y_pred_proba, save_path=pr_path)
    
    # Plot feature importance
    fi_path = os.path.join(output_dir, 'feature_importance.png')
    plot_feature_importance(top_features, top_importances, save_path=fi_path)
    
    print(f"✓ Visualisasi berhasil disimpan di: {output_dir}")
    
    # ============================================================
    # 8. SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("RINGKASAN HASIL")
    print("="*70)
    print(f"\n📊 PERFORMA MODEL:")
    print(f"  • Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  • Precision : {metrics['precision']:.4f}")
    print(f"  • Recall    : {metrics['recall']:.4f}")
    print(f"  • F1-Score  : {metrics['f1_score']:.4f}")
    print(f"  • ROC-AUC   : {metrics['roc_auc']:.4f}")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  • Model         : {model_save_path}")
    print(f"  • Confusion Matrix    : {cm_path}")
    print(f"  • ROC Curve           : {roc_path}")
    print(f"  • Precision-Recall    : {pr_path}")
    print(f"  • Feature Importance  : {fi_path}")
    
    print("\n" + "="*70 + "\n")
    

def predict_email(email_text):
    """
    Fungsi untuk memprediksi apakah email adalah spam atau bukan
    
    Parameters:
    -----------
    email_text : str
        Teks email yang akan diprediksi
    
    Returns:
    --------
    dict
        Dictionary berisi prediksi dan probabilitas
    """
    from model import load_model
    from data_preprocessing import clean_text
    
    # Load model
    model_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'models', 
        'spam_classifier_rf.pkl'
    )
    
    model = load_model(model_path)
    
    # Load vectorizer
    vectorizer_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'models', 
        'vectorizer.pkl'
    )
    
    # Clean text
    cleaned_text = clean_text(email_text)
    
    # Vectorize
    X = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    return {
        'email': email_text,
        'cleaned_text': cleaned_text,
        'prediction': 'SPAM' if prediction == 1 else 'HAM',
        'confidence': max(probability),
        'probability_ham': probability[0],
        'probability_spam': probability[1]
    }


if __name__ == "__main__":
    main()

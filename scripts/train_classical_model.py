# train_classical_models.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize, StandardScaler
from imblearn.over_sampling import ADASYN
import joblib
import re
import nltk
from nltk import pos_tag
from nltk.tokenize import WordPunctTokenizer

# -----------------------------
# Download NLTK resources
# -----------------------------
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger_eng")

# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "processed"))
PLOTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "plots"))
MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "models"))
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "master_data_vader.csv")
if not os.path.exists(DATA_PATH):
    print(f"❌ Dataset not found: {DATA_PATH}")
    exit()

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)
df = df[df['processed_text'].str.strip() != ""]
X_text = df['processed_text']
y = df['vader_sentiment']
classes = ["Positive", "Neutral", "Negative"]

# -----------------------------
# Feature Engineering
# -----------------------------
tokenizer = WordPunctTokenizer()

def extract_features(text_series):
    features = pd.DataFrame()
    # Text length
    features['text_len'] = text_series.str.len()
    # Exclamation and question mark counts
    features['excl_count'] = text_series.str.count('!')
    features['quest_count'] = text_series.str.count(r'\?')
    # VADER scores (already in dataset)
    features['vader_score'] = df['vader_score']

    nouns, adjectives = [], []
    for text in text_series:
        text = str(text) if pd.notna(text) else ""
        tokens = tokenizer.tokenize(text)
        pos_tags = pos_tag(tokens, lang='eng')
        nouns.append(len([w for w, t in pos_tags if t.startswith('NN')]))
        adjectives.append(len([w for w, t in pos_tags if t.startswith('JJ')]))
    features['noun_count'] = nouns
    features['adj_count'] = adjectives
    return features

X_features = extract_features(X_text)

# -----------------------------
# Train/test split
# -----------------------------
X_train_text, X_test_text, X_train_feat, X_test_feat, y_train, y_test = train_test_split(
    X_text, X_features, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# TF-IDF vectorization with selective ngrams
# -----------------------------
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,3),
                             min_df=2, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# -----------------------------
# Combine TF-IDF with engineered features
# -----------------------------
from scipy.sparse import hstack

scaler = StandardScaler()
X_train_feat_scaled = scaler.fit_transform(X_train_feat)
X_test_feat_scaled = scaler.transform(X_test_feat)

X_train_combined = hstack([X_train_tfidf, X_train_feat_scaled])
X_test_combined = hstack([X_test_tfidf, X_test_feat_scaled])

# -----------------------------
# Handle class imbalance with ADASYN
# -----------------------------
adasyn = ADASYN(random_state=42)
X_train_res, y_train_res = adasyn.fit_resample(X_train_combined, y_train)
print(f"🔹 Training set size after ADASYN: {X_train_res.shape[0]} rows")

# -----------------------------
# Define models with hyperparameter tweaks
# -----------------------------
lr = LogisticRegression(max_iter=1500, class_weight='balanced', random_state=42)
rf = RandomForestClassifier(n_estimators=300, max_features='sqrt',
                            min_samples_leaf=2, class_weight='balanced', random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(150,100), max_iter=800, alpha=0.001,
                    random_state=42)

# Stacking ensemble
stack = StackingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('mlp', mlp)],
    final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    cv=5,
    n_jobs=-1
)

models = {
    "logistic_regression": lr,
    "random_forest": rf,
    "mlp": mlp,
    "stacking_ensemble": stack
}

# -----------------------------
# Train, evaluate, save
# -----------------------------
for name, clf in models.items():
    print(f"\n🔹 Training {name}...")
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_test_combined)
    report = classification_report(y_test, y_pred)
    print(report)
    
    # Save report
    report_path = os.path.join(PLOTS_DIR, f"{name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{name}_confusion_matrix.png"))
    plt.close()

    # ROC & PR curves
    y_test_bin = label_binarize(y_test, classes=classes)
    y_pred_prob = clf.predict_proba(X_test_combined)

    # ROC
    plt.figure(figsize=(8,6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.title(f"{name} ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{name}_roc_curves.png"))
    plt.close()

    # PR curve
    plt.figure(figsize=(8,6))
    for i, cls in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_prob[:, i])
        plt.plot(recall, precision, label=cls)
    plt.title(f"{name} Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{name}_precision_recall_curves.png"))
    plt.close()

    # Save model
    model_path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
    joblib.dump(clf, model_path)

# Save TF-IDF vectorizer & scaler
joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "feature_scaler.pkl"))

print("\n✅ All models & vectorizer saved.")

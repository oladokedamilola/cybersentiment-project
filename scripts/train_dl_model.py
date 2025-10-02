# train_dl_model.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, LSTM, Bidirectional, Dropout,
    Input, Concatenate
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import random
import nltk
from nltk.corpus import wordnet

# -----------------------------
# Download NLTK WordNet
# -----------------------------
nltk.download('wordnet')
nltk.download('omw-1.4')

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
MODEL_PATH = os.path.join(MODELS_DIR, "cnn_bilstm_vader_multiinput.keras")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "dl_tokenizer.pkl")
REPORT_PATH = os.path.join(PLOTS_DIR, "dl_classification_report.txt")

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)
df['processed_text'] = df['processed_text'].fillna("")
vader_features = df[['vader_score']].values
print(f"📊 Loaded dataset with {len(df)} rows")

# -----------------------------
# Encode labels
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(df['vader_sentiment'])
num_classes = len(le.classes_)
y_categorical = to_categorical(y_encoded)

# -----------------------------
# Balance classes BEFORE split
# -----------------------------
df_balanced = df.copy()
df_balanced["label"] = y_encoded

neg = df_balanced[df_balanced["label"] == 0]
neu = df_balanced[df_balanced["label"] == 1]
pos = df_balanced[df_balanced["label"] == 2]

max_size = max(len(neg), len(neu), len(pos))
neg_upsampled = resample(neg, replace=True, n_samples=max_size, random_state=42)
neu_upsampled = resample(neu, replace=True, n_samples=max_size, random_state=42)
pos_upsampled = resample(pos, replace=True, n_samples=max_size, random_state=42)

df_balanced = pd.concat([neg_upsampled, neu_upsampled, pos_upsampled])
print(f"⚖️ Balanced dataset to {len(df_balanced)} samples")

# -----------------------------
# Train/test split
# -----------------------------
X_train_text, X_test_text, X_train_vader, X_test_vader, y_train, y_test = train_test_split(
    df_balanced['processed_text'], 
    df_balanced[['vader_score']], 
    to_categorical(df_balanced["label"], num_classes=num_classes),
    test_size=0.2, random_state=42, stratify=df_balanced["label"]
)
print(f"🔹 Training set: {len(X_train_text)} | Test set: {len(X_test_text)}")

# -----------------------------
# Tokenization & padding
# -----------------------------
MAX_VOCAB = 5000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=MAX_LEN, padding='post')
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_LEN, padding='post')

X_train_vader = np.array(X_train_vader)
X_test_vader = np.array(X_test_vader)

# -----------------------------
# Sentiment-protected words
# -----------------------------
SENTIMENT_WORDS = {
    'positive': ['good', 'great', 'love', 'excellent', 'awesome', 'happy', 'fantastic', 'amazing'],
    'negative': ['bad', 'terrible', 'hate', 'awful', 'worst', 'sad', 'horrible', 'poor'],
    'neutral': ['okay', 'fine', 'average', 'neutral', 'normal']
}
PROTECTED_WORDS = set(word for lst in SENTIMENT_WORDS.values() for word in lst)

# -----------------------------
# Augmentation functions
# -----------------------------
def synonym_replacement_safe(sentence, n=2):
    words = sentence.split()
    if not words: return sentence
    new_words = words.copy()
    random_indices = list(range(len(words)))
    random.shuffle(random_indices)
    replaced = 0
    for idx in random_indices:
        word = words[idx].lower()
        if word in PROTECTED_WORDS: continue
        synonyms = wordnet.synsets(word)
        if synonyms:
            lemmas = [l for l in synonyms[0].lemma_names() if l.lower() not in PROTECTED_WORDS]
            if lemmas:
                new_words[idx] = lemmas[0].replace("_", " ")
                replaced += 1
        if replaced >= n: break
    return " ".join(new_words)

def random_insertion_safe(sentence, n=1):
    words = sentence.split()
    if not words: return sentence
    for _ in range(n):
        idx = random.randint(0, len(words)-1)
        word = words[idx].lower()
        if word in PROTECTED_WORDS: continue
        synonyms = wordnet.synsets(word)
        if synonyms:
            lemmas = [l for l in synonyms[0].lemma_names() if l.lower() not in PROTECTED_WORDS]
            if lemmas:
                insert_idx = random.randint(0, len(words))
                words.insert(insert_idx, lemmas[0].replace("_", " "))
    return " ".join(words)

def random_swap_safe(sentence, n=1):
    words = sentence.split()
    if len(words) < 2: return sentence
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        if words[idx1].lower() in PROTECTED_WORDS or words[idx2].lower() in PROTECTED_WORDS: continue
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)

# -----------------------------
# Augment training data
# -----------------------------
AUGMENT_TIMES = 1
aug_texts, aug_vaders, aug_labels = [], [], []

for text, vader, label in zip(X_train_text, X_train_vader, y_train):
    aug_texts.append(text)
    aug_vaders.append(vader)
    aug_labels.append(label)
    for _ in range(AUGMENT_TIMES):
        choice = random.choice([synonym_replacement_safe, random_insertion_safe, random_swap_safe])
        aug_texts.append(choice(text))
        aug_vaders.append(vader)
        aug_labels.append(label)

X_train_pad = pad_sequences(tokenizer.texts_to_sequences(aug_texts), maxlen=MAX_LEN, padding='post')
X_train_vader = np.array(aug_vaders)
y_train = np.array(aug_labels)

# -----------------------------
# Compute class weights
# -----------------------------
y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight("balanced", classes=np.unique(y_integers), y=y_integers)
class_weights_dict = dict(enumerate(class_weights))
print(f"⚖️ Computed class weights: {class_weights_dict}")

# -----------------------------
# Multi-input Model (Text + VADER)
# -----------------------------
text_input = Input(shape=(MAX_LEN,), name="text_input")
x = Embedding(input_dim=MAX_VOCAB, output_dim=128, input_length=MAX_LEN)(text_input)
x = Conv1D(64, 5, activation='relu')(x)
x = Bidirectional(LSTM(32, return_sequences=True))(x)  # smaller LSTM
x = GlobalMaxPooling1D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.6)(x)

vader_input = Input(shape=(1,), name="vader_input")
v = Dense(16, activation='relu')(vader_input)

merged = Concatenate()([x, v])
out = Dense(num_classes, activation='softmax')(merged)

model = Model(inputs=[text_input, vader_input], outputs=out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# -----------------------------
# Callbacks
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)

# -----------------------------
# Train model
# -----------------------------
history = model.fit(
    [X_train_pad, X_train_vader], y_train,
    epochs=20,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights_dict
)

# -----------------------------
# Plots
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "dl_accuracy.png"), bbox_inches='tight')
plt.close()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "dl_loss.png"), bbox_inches='tight')
plt.close()

# -----------------------------
# Evaluate
# -----------------------------
y_pred_probs = model.predict([X_test_pad, X_test_vader])
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

report = classification_report(y_true, y_pred, target_names=le.classes_)
with open(REPORT_PATH, "w") as f:
    f.write(report)
print("\n📈 Classification Report:\n", report)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("DL Confusion Matrix")
plt.savefig(os.path.join(PLOTS_DIR, "dl_confusion_matrix.png"), bbox_inches='tight')
plt.close()

# -----------------------------
# Save model & tokenizer
# -----------------------------
model.save(MODEL_PATH)
joblib.dump(tokenizer, TOKENIZER_PATH)
print(f"\n✅ Saved CNN+BiLSTM model to {MODEL_PATH}")
print(f"✅ Saved tokenizer to {TOKENIZER_PATH}")

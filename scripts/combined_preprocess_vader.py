# combined_preprocess_vader.py
import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# -----------------------------
# Download required NLTK data
# -----------------------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")

# -----------------------------
# Directories
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "processed"))
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

print(f"📁 Processed data directory: {PROCESSED_DATA_DIR}")

# -----------------------------
# Text preprocessing tools
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')  # tokenize only words/numbers

def clean_text(text):
    """Clean and normalize text data"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

# -----------------------------
# Find the latest master CSV
# -----------------------------
all_master_csvs = [f for f in os.listdir(PROCESSED_DATA_DIR) if "master" in f and f.endswith(".csv")]
if not all_master_csvs:
    print("❌ No master CSV found in processed folder!")
    exit()

all_master_csvs.sort(key=lambda f: os.path.getmtime(os.path.join(PROCESSED_DATA_DIR, f)), reverse=True)
latest_master_csv = os.path.join(PROCESSED_DATA_DIR, all_master_csvs[0])
print(f"📄 Loading latest master CSV: {os.path.basename(latest_master_csv)}")

# -----------------------------
# Load and preprocess
# -----------------------------
try:
    df = pd.read_csv(latest_master_csv)
except Exception as e:
    print(f"❌ Failed to read {latest_master_csv}: {e}")
    exit()

if "text" not in df.columns:
    print(f"❌ 'text' column not found in {latest_master_csv}")
    exit()

print(f"📝 Cleaning {len(df)} rows...")
df["processed_text"] = df["text"].apply(clean_text)

# Drop empty rows after cleaning
df["processed_text"] = df["processed_text"].fillna("").astype(str)
df = df[df["processed_text"].str.strip() != ""]
df.reset_index(drop=True, inplace=True)
print(f"✅ Cleaned dataset: {len(df)} rows")

# -----------------------------
# Sentiment Analysis with VADER
# -----------------------------
sia = SentimentIntensityAnalyzer()
df["vader_score"] = df["processed_text"].apply(lambda x: sia.polarity_scores(str(x))["compound"])

def vader_label(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["vader_sentiment"] = df["vader_score"].apply(vader_label)

print("\n✅ VADER sentiment scores added.")
print(df[["processed_text", "vader_score", "vader_sentiment"]].head())

# -----------------------------
# Save final dataset
# -----------------------------
output_csv = os.path.join(PROCESSED_DATA_DIR, "master_data_vader.csv")
df.to_csv(output_csv, index=False)
print(f"\n✅ Final dataset saved to {output_csv}")

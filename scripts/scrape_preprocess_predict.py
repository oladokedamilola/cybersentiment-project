"""
scrape_preprocess_predict.py

Complete pipeline: Scrape from Twitter & Reddit, preprocess (clean + VADER), 
load classical artifacts (TF-IDF, scaler, logistic regression), predict sentiment & risk, 
persist Post rows and global Notifications/Alerts into Flask DB.

Designed to run in the same repository as your Flask app.
Run via: python -m webapp.scripts.scrape_preprocess_predict  (or directly)
"""
import os
import shutil
import time
import logging
import glob
from datetime import datetime, timezone
import re

import pandas as pd
import numpy as np
import joblib
import tweepy
import praw
import requests

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, WordPunctTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import pos_tag

from scipy.sparse import hstack

# -----------------------------
# Flask app context & models
# -----------------------------
from webapp.app import create_app, db

# Create the Flask app
app = create_app()

# Use the app context before importing models
with app.app_context():
    from webapp.models import User, Post, Alert, Notification
    
    # ✅ Now you can safely use db and models here
    # Example: quick test query
    print("Users in DB:", User.query.count())


# -----------------------------
# CONFIG / PATHS
# -----------------------------
import os

# Base directory → one level up from scripts/ (cybersentiment root)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data directories
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
INGESTED_DIR = os.path.join(RAW_DIR, "ingested")  
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Models directory
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Files
MASTER_RAW_CSV = os.path.join(RAW_DIR, "combined_data_master.csv")
MASTER_PROCESSED_CSV = os.path.join(PROCESSED_DIR, "master_data_vader.csv")

TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "logistic_regression_model.pkl")  # your selected model

# Ensure required directories exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(INGESTED_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------------
# Scraping Configuration
# -----------------------------
# Twitter keywords
TWITTER_KEYWORDS = ["ransomware", "phishing", "zero-day", "data breach", "cyberattack"]
TWITTER_MAX_RESULTS = 50

# Reddit configuration
REDDIT_KEYWORDS = [
    "ransomware", "phishing", "zero-day", "data breach", "cyberattack",
    "malware", "trojan", "spyware", "botnet", "vulnerability",
    "hack", "exploit", "credential leak", "phishing scam",
    "cybersecurity news", "cyber threats", "cyber incident",
    "ddos", "social engineering", "cyber espionage", "infosec", "privacy breach"
]

REDDIT_SUBREDDITS = [
    "cybersecurity", "netsec", "hacking", "Malware", 
    "ReverseEngineering", "Infosec", "Privacy", "ComputerSecurity",
    "TechNews", "SecurityNews"
]

REDDIT_LIMIT = 100  # Reduced for testing
REDDIT_HISTORICAL_START = 1609459200  # Jan 1, 2021

# -----------------------------
# DEBUG: print resolved paths
# -----------------------------
print("==== PATH CONFIGURATION ====")
print("BASE_DIR:", BASE_DIR)
print("RAW_DIR:", RAW_DIR)
print("PROCESSED_DIR:", PROCESSED_DIR)
print("INGESTED_DIR:", INGESTED_DIR)   
print("MODELS_DIR:", MODELS_DIR)
print("TFIDF_PATH:", TFIDF_PATH)
print("SCALER_PATH:", SCALER_PATH)
print("MODEL_PATH:", MODEL_PATH)
print("============================")

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scrape_predict")

# -----------------------------
# Ensure NLTK resources (download once; safe afterwards)
# -----------------------------
_nltk_packages = ["stopwords", "wordnet", "vader_lexicon", "punkt", "averaged_perceptron_tagger"]
for pkg in _nltk_packages:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg)
        except Exception:
            log.warning("Failed to download nltk package: %s", pkg)

# -----------------------------
# Text cleaning - same logic as pipeline
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r"\w+")

def clean_text(text: str) -> str:
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
# FIXED: Engineered features extractor (matches training exactly)
# -----------------------------
pos_tokenizer = WordPunctTokenizer()

def extract_engineered_features(series_text: pd.Series, vader_scores: pd.Series) -> pd.DataFrame:
    """Extract engineered features - EXACTLY matches training pipeline"""
    # Create DataFrame with proper column names like during training
    features = pd.DataFrame()
    
    # Text length
    features['text_len'] = series_text.str.len().fillna(0).astype(float)
    # Exclamation and question mark counts
    features['excl_count'] = series_text.str.count('!').fillna(0).astype(float)
    features['quest_count'] = series_text.str.count(r'\?').fillna(0).astype(float)
    # VADER scores
    features['vader_score'] = vader_scores.fillna(0).astype(float)

    nouns, adjectives = [], []
    for text in series_text.fillna("").astype(str):
        text = str(text) if pd.notna(text) else ""
        tokens = pos_tokenizer.tokenize(text)
        try:
            pos_tags = pos_tag(tokens, lang='eng')
        except Exception:
            pos_tags = [(w, "NN") for w in tokens]
        nouns.append(len([w for w, t in pos_tags if t.startswith('NN')]))
        adjectives.append(len([w for w, t in pos_tags if t.startswith('JJ')]))
    
    features['noun_count'] = nouns
    features['adj_count'] = adjectives
    
    return features

# -----------------------------
# Load artifacts
# -----------------------------
def load_artifacts():
    log.info("Loading TF-IDF vectorizer, scaler and model...")
    missing = [p for p in (TFIDF_PATH, SCALER_PATH, MODEL_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing artifact(s): {missing}")
    vectorizer = joblib.load(TFIDF_PATH)
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    log.info("Artifacts loaded.")
    return vectorizer, scaler, model

# -----------------------------
# Twitter Scraping Functions
# -----------------------------
def setup_twitter_client():
    """Initialize Twitter client with credentials"""
    from dotenv import load_dotenv
    load_dotenv()
    
    BEARER_TOKEN = os.getenv("TWITTER_BEARER")
    if not BEARER_TOKEN:
        raise ValueError("❌ Twitter Bearer Token not found. Please set TWITTER_BEARER in your .env file.")
    
    client = tweepy.Client(bearer_token=BEARER_TOKEN)
    return client

def fetch_tweets(client, query, max_results=50):
    """Fetch tweets matching the given query"""
    log.info(f"Searching for tweets with keyword: '{query}'")
    try:
        response = client.search_recent_tweets(
            query=query,
            tweet_fields=["created_at", "lang", "author_id", "public_metrics"],
            max_results=max_results
        )

        tweets = []
        if response.data:
            for tweet in response.data:
                tweets.append({
                    "platform": "Twitter",
                    "id": str(tweet.id),  # store as string for safety
                    "text": tweet.text,
                    "timestamp": tweet.created_at,
                    "author_id": str(tweet.author_id),
                    "likes": tweet.public_metrics["like_count"],
                    "retweets": tweet.public_metrics["retweet_count"],
                    "sentiment": None
                })
            log.info(f"Retrieved {len(tweets)} tweets for '{query}'.")
        else:
            log.info(f"No tweets found for '{query}'.")
        return tweets

    except tweepy.TooManyRequests as e:
        log.warning(f"Twitter API rate limit exceeded for '{query}': {e}")
        return []
    except Exception as e:
        log.error(f"Error fetching tweets for '{query}': {e}")
        return []

def scrape_twitter():
    """Scrape data from Twitter and return new tweets"""
    log.info("🚀 Starting Twitter scraper...")
    all_tweets = []
    
    try:
        client = setup_twitter_client()
        
        for word in TWITTER_KEYWORDS:
            results = fetch_tweets(client, word, max_results=TWITTER_MAX_RESULTS)
            all_tweets.extend(results)
            # Add small delay between requests to be respectful of API limits
            time.sleep(1)
            
        log.info(f"Twitter scraping completed: {len(all_tweets)} total tweets")
        return all_tweets
        
    except Exception as e:
        log.error(f"Twitter scraping failed: {e}")
        return []

# -----------------------------
# Reddit Scraping Functions
# -----------------------------
def setup_reddit_client():
    """Initialize Reddit client with credentials"""
    from dotenv import load_dotenv
    load_dotenv()
    
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )
    reddit.read_only = True
    assert reddit.read_only, "❌ Reddit authentication failed"
    return reddit

def fetch_reddit_recent(reddit, subreddit, keywords, limit=100, after=None):
    """Fetch recent Reddit posts via PRAW"""
    posts = []
    try:
        for submission in reddit.subreddit(subreddit).new(limit=limit):
            timestamp = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
            
            # FIXED: Handle after parameter properly
            if after is not None:
                # Ensure after is a datetime object for comparison
                if isinstance(after, str):
                    try:
                        after = pd.to_datetime(after).tz_localize(timezone.utc)
                    except:
                        # If parsing fails, skip timestamp filtering
                        pass
                if isinstance(after, (datetime, pd.Timestamp)) and timestamp <= after:
                    continue
                    
            title = submission.title.lower() if submission.title else ""
            body = submission.selftext.lower() if submission.selftext else ""
            if any(kw.lower() in title or kw.lower() in body for kw in keywords):
                posts.append({
                    "platform": "Reddit",
                    "id": submission.id,
                    "title": submission.title,
                    "text": submission.selftext,
                    "timestamp": timestamp,
                    "author": str(submission.author),
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "sentiment": None
                })
        log.info(f"Found {len(posts)} posts in r/{subreddit} (recent).")
    except Exception as e:
        log.error(f"Error fetching recent posts from r/{subreddit}: {e}")
    return posts

def fetch_reddit_historical(subreddit, keywords, start_epoch, end_epoch=None, size=100, retries=3):
    """Fetch historical Reddit posts via Pushshift"""
    url = "https://api.pushshift.io/reddit/search/submission/"
    all_posts = []
    params = {
        "subreddit": subreddit,
        "after": start_epoch,
        "before": end_epoch or int(time.time()),
        "size": size,
        "sort": "asc"
    }

    attempt = 0
    while True:
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json().get("data", [])
            if not data:
                break

            for post in data:
                title = post.get("title", "").lower()
                selftext = post.get("selftext", "").lower()
                if any(kw.lower() in title or kw.lower() in selftext for kw in keywords):
                    timestamp = datetime.fromtimestamp(post["created_utc"], tz=timezone.utc)
                    all_posts.append({
                        "platform": "Reddit",
                        "id": post["id"],
                        "title": post.get("title"),
                        "text": post.get("selftext"),
                        "timestamp": timestamp,
                        "author": post.get("author"),
                        "score": post.get("score"),
                        "num_comments": post.get("num_comments"),
                        "sentiment": None
                    })

            # Prepare for next batch
            last_ts = data[-1]["created_utc"]
            params["after"] = last_ts
            if len(data) < size:
                break  # No more posts

            attempt = 0  # reset retry counter on success

        except requests.exceptions.RequestException as e:
            attempt += 1
            if attempt > retries:
                log.error(f"Skipping r/{subreddit} after {retries} failed attempts: {e}")
                break
            wait = 5 * attempt
            log.warning(f"Pushshift request failed, retrying in {wait}s... ({attempt}/{retries})")
            time.sleep(wait)

    log.info(f"Retrieved {len(all_posts)} historical posts from r/{subreddit}.")
    return all_posts

def scrape_reddit():
    """Scrape data from Reddit and return new posts"""
    log.info("🚀 Starting Reddit scraper...")
    all_data = []
    last_timestamps = {}

    try:
        reddit = setup_reddit_client()

        # Load existing master CSV to determine last timestamps
        if os.path.exists(MASTER_RAW_CSV):
            try:
                master_df = pd.read_csv(MASTER_RAW_CSV, parse_dates=["timestamp"])
                # Get latest timestamp per subreddit
                for sub in REDDIT_SUBREDDITS:
                    sub_posts = master_df[(master_df["platform"] == "Reddit") & (master_df["id"].notna())]
                    if not sub_posts.empty:
                        last_timestamps[sub] = sub_posts["timestamp"].max()
                log.info(f"Loaded last timestamps for {len(last_timestamps)} subreddits")
            except Exception as e:
                log.warning(f"Could not load existing master CSV: {e}")
                master_df = pd.DataFrame()
        else:
            master_df = pd.DataFrame()
            log.info("No existing master CSV found")

        # Only fetch historical data if no existing data
        fetch_historical = master_df.empty

        for sub in REDDIT_SUBREDDITS:
            after_time = last_timestamps.get(sub)

            # Fetch historical posts for first-time run only
            if fetch_historical:
                log.info(f"Fetching historical data for r/{sub} (first run)")
                historical_posts = fetch_reddit_historical(sub, REDDIT_KEYWORDS, start_epoch=REDDIT_HISTORICAL_START, size=50)
                if historical_posts:
                    all_data.extend(historical_posts)
                    if historical_posts:
                        last_timestamps[sub] = max(p["timestamp"] for p in historical_posts)

            # Always fetch recent posts
            log.info(f"Fetching recent posts for r/{sub}")
            recent_posts = fetch_reddit_recent(reddit, sub, REDDIT_KEYWORDS, limit=REDDIT_LIMIT, after=after_time)
            if recent_posts:
                all_data.extend(recent_posts)
                if recent_posts:
                    last_timestamps[sub] = max(p["timestamp"] for p in recent_posts)
            
            # Small delay between subreddits to be respectful
            time.sleep(1)

        log.info(f"Reddit scraping completed: {len(all_data)} total posts")
        return all_data
        
    except Exception as e:
        log.error(f"Reddit scraping failed: {e}")
        return []

# -----------------------------
# Main Scraping Function
# -----------------------------
def run_scrapers():
    """Run both Twitter and Reddit scrapers and save to individual CSV files"""
    log.info("Starting data scraping from Twitter and Reddit...")
    
    # Scrape from both platforms
    twitter_data = scrape_twitter()
    reddit_data = scrape_reddit()
    
    all_data = twitter_data + reddit_data
    new_files = []
    
    if all_data:
        # Create timestamped CSV file for this scraping session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scraped_csv = os.path.join(RAW_DIR, f"scraped_data_{timestamp}.csv")
        
        # Save scraped data to new CSV
        df = pd.DataFrame(all_data)
        df.to_csv(scraped_csv, index=False)
        new_files.append(scraped_csv)
        
        log.info(f"Scraping completed: {len(twitter_data)} tweets, {len(reddit_data)} Reddit posts")
        log.info(f"Saved to: {scraped_csv}")
    else:
        log.info("No new data scraped from either platform")
    
    return new_files

# -----------------------------
# Append new raw CSVs into a single master CSV.
# - Looks for any CSV files in RAW_DIR that are not the MASTER_RAW_CSV
# - Appends new unique rows (dedupe by platform+id when available)
# - Moves ingested raw files into RAW_DIR/ingested/
# -----------------------------
def update_master_raw_csv():
    raw_csvs = [
        p for p in glob.glob(os.path.join(RAW_DIR, "*.csv"))
        if os.path.basename(p) != os.path.basename(MASTER_RAW_CSV)
    ]
    if not raw_csvs:
        log.info("No new raw CSV files to ingest.")
        return []

    log.info("Found %d raw CSV(s) to ingest.", len(raw_csvs))
    frames = []
    for p in raw_csvs:
        try:
            df = pd.read_csv(p)
            frames.append(df)
        except Exception as e:
            log.warning("Failed to read %s: %s", p, e)

    if not frames:
        log.info("No valid CSV frames to append.")
        return []

    new_df = pd.concat(frames, ignore_index=True)
    if new_df.empty:
        log.info("Combined new CSVs produced no rows.")
        # move files to ingested anyway to avoid reprocessing corrupt or empty files
        for p in raw_csvs:
            shutil.move(p, os.path.join(INGESTED_DIR, os.path.basename(p)))
        return []

    # If master exists, load and append; otherwise create new master
    if os.path.exists(MASTER_RAW_CSV):
        master_df = pd.read_csv(MASTER_RAW_CSV)
        combined = pd.concat([master_df, new_df], ignore_index=True)
    else:
        combined = new_df

    # Deduplicate by platform+id if present, otherwise by all columns
    if "platform" in combined.columns and "id" in combined.columns:
        combined["id"] = combined["id"].astype(str)
        combined["platform"] = combined["platform"].astype(str)
        combined.drop_duplicates(subset=["platform", "id"], inplace=True, keep="last")
    else:
        combined.drop_duplicates(subset=list(combined.columns), inplace=True)

    combined.to_csv(MASTER_RAW_CSV, index=False)
    log.info("Master raw CSV updated at %s (%d rows)", MASTER_RAW_CSV, len(combined))

    # Move ingested raw files into ingested folder
    for p in raw_csvs:
        dest = os.path.join(INGESTED_DIR, os.path.basename(p))
        try:
            shutil.move(p, dest)
        except Exception as e:
            log.warning("Could not move %s to %s: %s", p, dest, e)
    
    return raw_csvs  # Return the list of processed files

# -----------------------------
# Preprocess master -> processed (clean + VADER)
# -----------------------------
def create_processed_master():
    if not os.path.exists(MASTER_RAW_CSV):
        log.info("No master raw CSV to preprocess.")
        return None

    log.info("Loading master raw CSV for preprocessing...")
    df = pd.read_csv(MASTER_RAW_CSV)
    if "text" not in df.columns:
        log.error("Master raw CSV has no 'text' column.")
        return None

    log.info("Cleaning text (%d rows)...", len(df))
    df["processed_text"] = df["text"].apply(clean_text)
    df["processed_text"] = df["processed_text"].fillna("").astype(str)
    df = df[df["processed_text"].str.strip() != ""].copy()
    df.reset_index(drop=True, inplace=True)
    log.info("After cleaning: %d rows", len(df))

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

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(MASTER_PROCESSED_CSV, index=False)
    log.info("Saved processed master CSV: %s (%d rows)", MASTER_PROCESSED_CSV, len(df))
    return df

# -----------------------------
# Helper: safe parse timestamp to timezone-aware datetime
# -----------------------------
def parse_timestamp(val):
    if pd.isna(val):
        return datetime.now(timezone.utc)
    try:
        # If pandas already parsed, it may be Timestamp
        if isinstance(val, (pd.Timestamp, datetime)):
            ts = pd.to_datetime(val)
        else:
            ts = pd.to_datetime(str(val))
        if ts.tzinfo is None:
            ts = ts.tz_localize(timezone.utc)
        return ts.to_pydatetime()
    except Exception:
        return datetime.now(timezone.utc)

# -----------------------------
# Main: process rows, predict and save to DB
# -----------------------------
def process_and_save():
    with app.app_context():   # ✅ reuse the global app
        vectorizer, scaler, model = load_artifacts()

        # Step 0: Run scrapers to get fresh data
        log.info("=== STEP 0: Data Scraping ===")
        scraped_files = run_scrapers()

        # Step 1: Ingest new CSVs into master
        log.info("=== STEP 1: Data Ingestion ===")
        new_files = update_master_raw_csv()

        # Step 2: Preprocess into processed master
        log.info("=== STEP 2: Data Preprocessing ===")
        df = create_processed_master()
        if df is None or df.empty:
            log.info("No processed rows to handle.")
            return

        # create external_id column for deduplication (platform + id)
        df["external_id"] = df.apply(lambda r: f"{r.get('platform','')}_{r.get('id','')}", axis=1)

        # Preload existing external_ids to avoid duplicates (if Post.external_id exists)
        existing_ids = set()
        if hasattr(Post, "external_id"):
            q = db.session.query(Post.external_id).filter(Post.external_id.isnot(None))
            for (eid,) in q:
                existing_ids.add(str(eid))

        processed_count = 0
        notif_count = 0
        alert_count = 0

        log.info("=== STEP 3: Prediction & Database Storage ===")
        for idx, row in df.iterrows():
            try:
                external_id = str(row.get("external_id", ""))

                # Skip duplicates if master already in DB
                if external_id and external_id in existing_ids:
                    continue

                # If Post has no external_id, fallback duplication check
                if not external_id and Post.query.filter_by(text=row.get("text", "")).first():
                    continue

                proc_text = row.get("processed_text", "")
                # TF-IDF
                tfidf_vec = vectorizer.transform([proc_text])

                # FIXED: engineered features - now returns DataFrame with proper column names
                feat_df = extract_engineered_features(
                    pd.Series([row.get("text", "")]),
                    pd.Series([row.get("vader_score", 0.0)])
                )
                # Now scaler gets proper DataFrame with column names
                feat_scaled = scaler.transform(feat_df)
                X_comb = hstack([tfidf_vec, feat_scaled])

                # predict
                pred_label = model.predict(X_comb)[0]
                prob = model.predict_proba(X_comb)[0] if hasattr(model, "predict_proba") else None

                # risk policy
                risk_flag = False
                if pred_label == "Negative":
                    risk_flag = True
                elif prob is not None and hasattr(model, "classes_"):
                    try:
                        class_index = list(model.classes_).index("Negative")
                        if prob[class_index] >= 0.6:
                            risk_flag = True
                    except ValueError:
                        pass

                ts = parse_timestamp(row.get("timestamp", None))

                # Build Post object
                post_kwargs = {
                    "text": row.get("text", ""),
                    "predicted_sentiment": pred_label,
                    "risk_flag": bool(risk_flag),
                    "timestamp": ts,
                }
                if hasattr(Post, "processed_text"):
                    post_kwargs["processed_text"] = proc_text
                if hasattr(Post, "vader_sentiment"):
                    post_kwargs["vader_sentiment"] = row.get("vader_sentiment")
                if hasattr(Post, "vader_score"):
                    try:
                        post_kwargs["vader_score"] = float(row.get("vader_score", 0.0))
                    except Exception:
                        post_kwargs["vader_score"] = None
                if hasattr(Post, "ml_probability"):
                    try:
                        post_kwargs["ml_probability"] = float(np.max(prob)) if prob is not None else None
                    except Exception:
                        post_kwargs["ml_probability"] = None
                if hasattr(Post, "external_id"):
                    post_kwargs["external_id"] = external_id or None
                if hasattr(Post, "platform"):
                    post_kwargs["platform"] = row.get("platform", None)

                post = Post(**post_kwargs)
                db.session.add(post)
                db.session.flush()
                processed_count += 1
                if external_id:
                    existing_ids.add(external_id)

                # If risky, create a notification AND an alert
                if risk_flag:
                    message = f"Risky post detected ({row.get('platform','unknown')}) id={row.get('id')}"
                    notif_kwargs = {
                        "message": message,
                        "timestamp": datetime.now(timezone.utc),
                    }
                    try:
                        notif = Notification(**notif_kwargs)
                        if hasattr(notif, "post_id"):
                            notif.post_id = post.id
                        db.session.add(notif)
                        notif_count += 1
                    except Exception as e:
                        log.warning("Could not create Notification object: %s", e)

                    # NEW: Also create a global Alert for flagged posts
                    try:
                        alert = Alert(
                            post_id=str(post.id),  # Using the internal post ID
                            reason=f"Automated risk detection: {pred_label} sentiment with high risk probability",
                            created_at=datetime.now(timezone.utc)
                        )
                        db.session.add(alert)
                        alert_count += 1
                        log.info("Created global alert for risky post ID: %s", post.id)
                    except Exception as e:
                        log.warning("Could not create Alert object: %s", e)

            except Exception as ex:
                log.exception("Failed processing row idx=%s: %s", idx, ex)
                db.session.rollback()
                continue

        try:
            db.session.commit()
        except Exception as e:
            log.exception("DB commit failed: %s", e)
            db.session.rollback()

        # Step 3: Move processed raw files into ingested archive (with better error handling)
        if new_files:
            for p in new_files:
                if os.path.exists(p):  # Check if file still exists
                    dest = os.path.join(INGESTED_DIR, os.path.basename(p))
                    if os.path.exists(dest):
                        base, ext = os.path.splitext(dest)
                        dest = f"{base}_{int(time.time())}{ext}"
                    try:
                        shutil.move(p, dest)
                        log.info(f"Moved {os.path.basename(p)} to ingested folder")
                    except Exception as e:
                        log.warning(f"Could not move {p} to {dest}: {e}")
                else:
                    log.info(f"File already processed or moved: {p}")

        # Step 4: Log summary - FIXED: Handle file reading errors gracefully
        try:
            raw_master_count = 0
            processed_master_count = 0
            
            if os.path.exists(MASTER_RAW_CSV):
                raw_master = pd.read_csv(MASTER_RAW_CSV)
                raw_master_count = len(raw_master)
            
            if os.path.exists(MASTER_PROCESSED_CSV):
                processed_master = pd.read_csv(MASTER_PROCESSED_CSV)
                processed_master_count = len(processed_master)
            
            log.info("SUMMARY: Scraped %d files | Ingested %d files | Master rows=%d | Processed rows=%d | Saved posts=%d | Notifications=%d | Alerts=%d",
                     len(scraped_files), len(new_files), raw_master_count, processed_master_count, processed_count, notif_count, alert_count)
        except Exception as e:
            log.warning("Could not generate summary: %s", e)

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    t0 = time.time()
    try:
        log.info("Starting complete scrape_preprocess_predict pipeline")
        process_and_save()
    except Exception as e:
        log.exception("Top-level error: %s", e)
    finally:
        log.info("Finished. Elapsed: %.1fs", time.time() - t0)
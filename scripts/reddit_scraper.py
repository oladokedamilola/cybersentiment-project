# continuous_scraper_reddit_historical.py
import os
import time
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv
import praw
import requests

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Directories
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "raw"))
os.makedirs(RAW_DATA_DIR, exist_ok=True)
MASTER_CSV = os.path.join(RAW_DATA_DIR, "combined_data_master.csv")
print(f"📁 Raw data directory: {RAW_DATA_DIR}")

# -----------------------------
# Expanded keywords and subreddits
# -----------------------------
KEYWORDS = [
    "ransomware", "phishing", "zero-day", "data breach", "cyberattack",
    "malware", "trojan", "spyware", "botnet", "vulnerability",
    "hack", "exploit", "credential leak", "phishing scam",
    "cybersecurity news", "cyber threats", "cyber incident",
    "ddos", "social engineering", "cyber espionage", "infosec", "privacy breach"
]

SUBREDDITS = [
    "cybersecurity", "netsec", "hacking", "Malware", 
    "ReverseEngineering", "Infosec", "Privacy", "ComputerSecurity",
    "TechNews", "SecurityNews"
]

# -----------------------------
# Reddit setup (read-only)
# -----------------------------
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)
reddit.read_only = True
assert reddit.read_only, "❌ Reddit authentication failed"

# -----------------------------
# Fetch recent posts via PRAW
# -----------------------------
def fetch_reddit(subreddit, keywords, limit=500, after=None):
    posts = []
    try:
        for submission in reddit.subreddit(subreddit).new(limit=limit):
            timestamp = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
            if after and timestamp <= after:
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
        print(f"✅ Found {len(posts)} posts in r/{subreddit} (recent).")
    except Exception as e:
        print(f"❌ Error fetching recent posts from r/{subreddit}: {e}")
    return posts

# -----------------------------
# Fetch historical posts via Pushshift with retry/backoff
# -----------------------------
def fetch_reddit_historical(subreddit, keywords, start_epoch, end_epoch=None, size=500, retries=3):
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
                print(f"❌ Skipping r/{subreddit} after {retries} failed attempts: {e}")
                break
            wait = 5 * attempt
            print(f"⚠️ Pushshift request failed, retrying in {wait}s... ({attempt}/{retries})")
            time.sleep(wait)

    print(f"✅ Retrieved {len(all_posts)} historical posts from r/{subreddit}.")
    return all_posts

# -----------------------------
# Continuous scraping loop
# -----------------------------
FETCH_INTERVAL = 60 * 30  # 30 minutes
last_timestamps = {}  # track last post per subreddit

while True:
    print(f"\n🚀 Starting Reddit data fetch at {datetime.now(timezone.utc)} ...")
    all_data = []

    # Load existing master CSV
    if os.path.exists(MASTER_CSV):
        master_df = pd.read_csv(MASTER_CSV, parse_dates=["timestamp"])
    else:
        master_df = pd.DataFrame()

    for sub in SUBREDDITS:
        after_time = last_timestamps.get(sub)

        # Fetch historical posts for first-time run
        if after_time is None:
            historical_posts = fetch_reddit_historical(sub, KEYWORDS, start_epoch=1609459200)  # Jan 1, 2021
            if historical_posts:
                all_data.extend(historical_posts)
                last_timestamps[sub] = max(p["timestamp"] for p in historical_posts)

        # Fetch recent posts
        recent_posts = fetch_reddit(sub, KEYWORDS, limit=500, after=last_timestamps.get(sub))
        if recent_posts:
            all_data.extend(recent_posts)
            last_timestamps[sub] = max(p["timestamp"] for p in recent_posts)

    # Append new data to master CSV, avoiding duplicates
    if all_data:
        new_df = pd.DataFrame(all_data)
        combined_df = pd.concat([master_df, new_df]).drop_duplicates(subset=["platform", "id"])
        combined_df.to_csv(MASTER_CSV, index=False)
        print(f"📁 Master CSV updated: {len(combined_df)} total records")
    else:
        print("⚠️ No new data collected this cycle")

    print(f"⏳ Sleeping {FETCH_INTERVAL / 60} minutes before next fetch...\n")
    time.sleep(FETCH_INTERVAL)

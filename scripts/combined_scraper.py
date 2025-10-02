# continuous_scraper_x_reddit_fallback.py
import os
import time
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import tweepy
import praw

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
# Keywords and subreddits
# -----------------------------
KEYWORDS = ["ransomware", "phishing", "zero-day", "data breach", "cyberattack"]
SUBREDDITS = ["cybersecurity", "netsec", "hacking"]

# -----------------------------
# Twitter/X setup (read-only)
# -----------------------------
BEARER_TOKEN = os.getenv("TWITTER_BEARER")
if not BEARER_TOKEN:
    raise ValueError("❌ Twitter Bearer Token not found. Set TWITTER_BEARER in .env.")

# Read-only client: can only fetch data
twitter_client = tweepy.Client(bearer_token=BEARER_TOKEN)

def fetch_tweets(query, max_total=200):
    """Fetch up to max_total tweets with pagination (read-only)"""
    tweets = []
    next_token = None
    remaining = max_total

    while remaining > 0:
        batch_size = min(100, remaining)
        try:
            response = twitter_client.search_recent_tweets(
                query=query,
                tweet_fields=["created_at", "lang", "author_id", "public_metrics"],
                max_results=batch_size,
                next_token=next_token
            )
            if not response.data:
                break

            for tweet in response.data:
                tweets.append({
                    "platform": "Twitter",
                    "id": tweet.id,
                    "text": tweet.text,
                    "timestamp": tweet.created_at,
                    "author": tweet.author_id,
                    "likes": tweet.public_metrics["like_count"],
                    "retweets": tweet.public_metrics["retweet_count"],
                    "sentiment": None
                })

            remaining -= len(response.data)
            next_token = response.meta.get("next_token")
            if not next_token:
                break

        except tweepy.TooManyRequests:
            print("⚠️ Twitter/X rate limit hit. Falling back to Reddit...")
            return None  # signal to switch to Reddit
        except Exception as e:
            print(f"❌ Error fetching tweets '{query}': {e}")
            break

    print(f"✅ Retrieved {len(tweets)} tweets for '{query}'.")
    return tweets

# -----------------------------
# Reddit setup (read-only)
# -----------------------------
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)
reddit.read_only = True  # ensure read-only access
assert reddit.read_only, "❌ Reddit authentication failed"

def fetch_reddit(subreddit, keywords, limit=200):
    posts = []
    try:
        for submission in reddit.subreddit(subreddit).new(limit=limit):
            title = submission.title.lower() if submission.title else ""
            body = submission.selftext.lower() if submission.selftext else ""
            if any(word in title or word in body for word in keywords):
                posts.append({
                    "platform": "Reddit",
                    "id": submission.id,
                    "title": submission.title,
                    "text": submission.selftext,
                    "timestamp": datetime.utcfromtimestamp(submission.created_utc),
                    "author": str(submission.author),
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "sentiment": None
                })
        print(f"✅ Found {len(posts)} posts in r/{subreddit}.")
    except Exception as e:
        print(f"❌ Error fetching from r/{subreddit}: {e}")
    return posts

# -----------------------------
# Continuous scraping loop
# -----------------------------
FETCH_INTERVAL = 60 * 60 * 2  # fetch every 2 hours

while True:
    print(f"\n🚀 Starting data fetch at {datetime.now()} ...")
    all_data = []

    for kw in KEYWORDS:
        # Try Twitter/X first (read-only)
        tweets = fetch_tweets(kw, max_total=200)
        if tweets is None:
            # Twitter rate limit hit → fallback to Reddit for this keyword
            for sub in SUBREDDITS:
                all_data.extend(fetch_reddit(sub, [kw], limit=200))
        else:
            all_data.extend(tweets)

    # Load existing master CSV if exists
    if os.path.exists(MASTER_CSV):
        master_df = pd.read_csv(MASTER_CSV)
    else:
        master_df = pd.DataFrame()

    # Append new data, avoid duplicates
    new_df = pd.DataFrame(all_data)
    if not new_df.empty:
        combined_df = pd.concat([master_df, new_df]).drop_duplicates(subset=["platform", "id"])
        combined_df.to_csv(MASTER_CSV, index=False)
        print(f"📁 Master CSV updated: {len(combined_df)} total records")
    else:
        print("⚠️ No new data collected this cycle")

    print(f"⏳ Sleeping {FETCH_INTERVAL / 3600} hours before next fetch...\n")
    time.sleep(FETCH_INTERVAL)

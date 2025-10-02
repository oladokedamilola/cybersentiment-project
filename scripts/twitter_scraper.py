# twitter_scraper.py
import os
import tweepy
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Directories (relative to this script)
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "raw"))
os.makedirs(RAW_DATA_DIR, exist_ok=True)

MASTER_CSV = os.path.join(RAW_DATA_DIR, "combined_data_master.csv")
print(f"📁 Master data file: {MASTER_CSV}")

# -----------------------------
# Grab credentials
# -----------------------------
BEARER_TOKEN = os.getenv("TWITTER_BEARER")
if not BEARER_TOKEN:
    raise ValueError("❌ Twitter Bearer Token not found. Please set TWITTER_BEARER in your .env file.")

# Set up authentication
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# -----------------------------
# Define keywords
# -----------------------------
keywords = ["ransomware", "phishing", "zero-day", "data breach", "cyberattack"]

# -----------------------------
# Fetch tweets function
# -----------------------------
def fetch_tweets(query, max_results=50):
    """Fetch tweets matching the given query"""
    print(f"\n🔍 Searching for tweets with keyword: '{query}' ...")
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
                    "sentiment": None  # placeholder
                })
            print(f"✅ Retrieved {len(tweets)} tweets for '{query}'.")
        else:
            print(f"⚠️ No tweets found for '{query}'.")
        return tweets

    except Exception as e:
        print(f"❌ Error fetching tweets for '{query}': {e}")
        return []

# -----------------------------
# Append to master CSV
# -----------------------------
def update_master_csv(new_data):
    """Append new rows to master CSV while avoiding duplicates"""
    new_df = pd.DataFrame(new_data)

    if os.path.exists(MASTER_CSV):
        master_df = pd.read_csv(MASTER_CSV)

        # Ensure same dtypes
        master_df["id"] = master_df["id"].astype(str)

        # Avoid duplicates by keeping only new IDs
        before_count = len(master_df)
        combined_df = pd.concat([master_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset=["platform", "id"], inplace=True)
        after_count = len(combined_df)

        combined_df.to_csv(MASTER_CSV, index=False)
        print(f"📊 Master file updated. Added {after_count - before_count} new rows.")
    else:
        # First time creating master file
        new_df.to_csv(MASTER_CSV, index=False)
        print(f"📊 Master file created with {len(new_df)} rows.")

# -----------------------------
# Main script
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting Twitter scraper...\n")
    all_tweets = []

    for word in keywords:
        results = fetch_tweets(word, max_results=50)
        all_tweets.extend(results)

    if all_tweets:
        update_master_csv(all_tweets)
    else:
        print("\n⚠️ No tweets collected. Try adjusting keywords or increasing max_results.")

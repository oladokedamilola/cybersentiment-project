from flask import Blueprint, render_template, request
from flask_login import login_required
from webapp.models import Post
from datetime import datetime, timedelta
from collections import defaultdict
import json

analytics_bp = Blueprint("analytics", __name__, url_prefix="/analytics")

@analytics_bp.route("/sentiment-trends")
@login_required
def sentiment_trends():
    # --- Filters ---
    platform = request.args.get("platform", "").lower()
    try:
        days = int(request.args.get("days", 30))
    except ValueError:
        days = 30

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # --- Query ---
    query = Post.query.filter(Post.timestamp >= start_date, Post.timestamp <= end_date)
    if platform in ["twitter", "reddit"]:
        query = query.filter(Post.platform == platform)

    posts = query.all()

    # --- Aggregate by date ---
    counts = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0})
    for post in posts:
        if not post.sentiment:
            continue
        day = post.timestamp.strftime("%Y-%m-%d")
        counts[day][post.sentiment] += 1

    # --- Ensure continuous date range (for cleaner charts) ---
    date_range = [ (start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days+1) ]

    positive_counts = [counts[date]["positive"] for date in date_range]
    neutral_counts = [counts[date]["neutral"] for date in date_range]
    negative_counts = [counts[date]["negative"] for date in date_range]

    return render_template(
        "sentiment_trends.html",
        platform=platform,
        days=days,
        dates=json.dumps(date_range),
        positive_counts=json.dumps(positive_counts),
        neutral_counts=json.dumps(neutral_counts),
        negative_counts=json.dumps(negative_counts),
        total_posts=len(posts),
        last_updated=end_date.strftime("%b %d, %Y %H:%M UTC"),
    )

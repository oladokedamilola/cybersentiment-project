from flask import Blueprint, render_template
from flask_login import login_required, current_user
from datetime import datetime
from sqlalchemy import func, desc

from webapp.models import Post, Alert, Notification, NotificationRead

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/dashboard")

@dashboard_bp.route("/")
@login_required
def index():
    # -----------------------------
    # Sentiment counts (from Posts table)
    # -----------------------------
    sentiment_counts = (
        Post.query.with_entities(Post.predicted_sentiment, func.count(Post.id))
        .group_by(Post.predicted_sentiment)
        .all()
    )
    sentiment_counts_dict = {s.lower(): c for s, c in sentiment_counts if s}

    # fill defaults so template doesn't break
    sentiment_counts = {
        "positive": sentiment_counts_dict.get("positive", 0),
        "neutral": sentiment_counts_dict.get("neutral", 0),
        "negative": sentiment_counts_dict.get("negative", 0),
    }

    # -----------------------------
    # Latest alerts (last 5)
    # -----------------------------
    latest_alerts = (
        Alert.query.order_by(Alert.timestamp.desc())
        .limit(5)
        .all()
    )

    # -----------------------------
    # Last updated = most recent post timestamp
    # -----------------------------
    last_post = Post.query.order_by(Post.timestamp.desc()).first()
    last_updated = last_post.timestamp if last_post else datetime.now()

    # -----------------------------
    # Unread notifications for current user - FIXED
    # -----------------------------
    # Get IDs of notifications the user has read
    read_ids = {nr.notification_id for nr in current_user.notification_reads}
    
    # Get unread notifications (not in read_ids)
    unread_notifications = Notification.query.filter(
        ~Notification.id.in_(read_ids)
    ).order_by(Notification.timestamp.desc()).limit(5).all()

    return render_template(
        "dashboard.html",
        sentiment_counts=sentiment_counts,
        latest_alerts=latest_alerts,
        last_updated=last_updated,
        unread_notifications=unread_notifications
    )
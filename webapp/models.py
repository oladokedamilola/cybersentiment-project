from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from webapp.app import db, login_manager


# --- User Model ---
class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(25), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    first_name = db.Column(db.String(30), nullable=True)
    last_name = db.Column(db.String(30), nullable=True)
    profile_image = db.Column(db.String(255), nullable=True)  # store filename/path
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships - FIXED: Remove duplicate backref names
    alerts = db.relationship("Alert", backref="alert_user", lazy=True)  # Changed backref name
    posts = db.relationship("Post", backref="author", lazy=True)
    
    # Alert reads relationship
    alert_reads = db.relationship(  # Renamed from read_alerts
        "AlertRead", 
        back_populates="user",
        lazy=True
    )
    
    # Notification reads relationship
    notification_reads = db.relationship(
        "NotificationRead", 
        back_populates="user",
        lazy=True
    )

    def set_password(self, password: str):
        """Hashes and sets the user's password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Validates the user's password."""
        return check_password_hash(self.password_hash, password)
    
    def get_full_name(self):
        """Return first + last name if available, else username."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        else:
            return self.username

    def __repr__(self):
        return f"<User {self.username}>"


# --- Alert Model ---
class Alert(db.Model):
    __tablename__ = "alerts"
    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(db.String(50), nullable=False)  # Changed back to String for external IDs
    reason = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)

    def __repr__(self):
        return f"<Alert {self.post_id}: {self.reason}>"


class AlertRead(db.Model):
    __tablename__ = "alert_reads"
    id = db.Column(db.Integer, primary_key=True)
    alert_id = db.Column(db.Integer, db.ForeignKey('alerts.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    read_at = db.Column(db.DateTime, nullable=True)
    
    # Constraints
    __table_args__ = (db.UniqueConstraint('alert_id', 'user_id', name='uq_alert_user'),)
    
    # Relationships with back_populates
    user = db.relationship("User", back_populates="alert_reads")  # Updated to match User model
    alert = db.relationship("Alert", backref="read_statuses")


class Post(db.Model):
    __tablename__ = "posts"
    id = db.Column(db.Integer, primary_key=True)
    external_id = db.Column(db.String(120), unique=True, index=True, nullable=True)  # platform_id combo, unique
    platform = db.Column(db.String(30), nullable=True)
    title = db.Column(db.String(500), nullable=True)
    text = db.Column(db.Text, nullable=False)
    processed_text = db.Column(db.Text, nullable=True)
    vader_score = db.Column(db.Float, nullable=True)
    vader_sentiment = db.Column(db.String(20), nullable=True)
    predicted_sentiment = db.Column(db.String(20), nullable=True)
    ml_probability = db.Column(db.Float, nullable=True)  # max predicted prob
    risk_flag = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # optional: who scraped / external owner
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    def __repr__(self):
        return f"<Post {self.id}: {self.predicted_sentiment}>"


class Notification(db.Model):
    __tablename__ = "notifications"
    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(db.Integer, db.ForeignKey('posts.id'), nullable=True)
    message = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow) 
    
    # Relationships
    post = db.relationship("Post", backref="notifications")
    read_statuses = db.relationship(
        "NotificationRead", 
        back_populates="notification",
        lazy=True
    )


class NotificationRead(db.Model):
    __tablename__ = "notification_reads"
    id = db.Column(db.Integer, primary_key=True)
    notification_id = db.Column(db.Integer, db.ForeignKey('notifications.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    read_at = db.Column(db.DateTime, nullable=True)
    
    # Constraints
    __table_args__ = (db.UniqueConstraint('notification_id', 'user_id', name='uq_notif_user'),)
    
    # Relationships with back_populates
    user = db.relationship(
        "User", 
        back_populates="notification_reads"
    )
    notification = db.relationship(
        "Notification", 
        back_populates="read_statuses"
    )


# --- Flask-Login User Loader ---
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
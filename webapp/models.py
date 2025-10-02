from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from webapp.app import db, login_manager
from datetime import datetime


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

    # Relationship: one user can have many alerts
    alerts = db.relationship("Alert", backref="user", lazy=True)
    notifications = db.relationship("NotificationRead", backref="user", lazy=True)
    read_notifications = db.relationship("NotificationRead", backref="read_by_user", lazy=True)

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
    post_id = db.Column(db.String(50), nullable=False)
    reason = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # Foreign key to link alert to user
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    def __repr__(self):
        return f"<Alert {self.post_id}: {self.reason}>"

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
        return f"<Post {self.id}: {self.predicted_sentiment}>"  # Fixed this too


# --- Flask-Login User Loader ---
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class Notification(db.Model):
    __tablename__ = "notifications"
    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(db.Integer, db.ForeignKey('posts.id'), nullable=True)
    message = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # optional
    # relationship
    post = db.relationship("Post", backref="notifications")

class NotificationRead(db.Model):
    __tablename__ = "notification_reads"
    id = db.Column(db.Integer, primary_key=True)
    notification_id = db.Column(db.Integer, db.ForeignKey('notifications.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    read_at = db.Column(db.DateTime, nullable=True)
    # constraints
    __table_args__ = (db.UniqueConstraint('notification_id', 'user_id', name='uq_notif_user'),)
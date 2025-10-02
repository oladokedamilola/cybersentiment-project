from flask import current_app
from flask_login import current_user
from webapp.models import Notification, NotificationRead

def unread_notifications():
    if current_user.is_authenticated:
        # Get IDs of notifications the user has read
        read_ids = {nr.notification_id for nr in current_user.read_notifications}
        # Query all notifications not in that set
        unread = Notification.query.filter(~Notification.id.in_(read_ids)).order_by(Notification.created_at.desc()).all()
        return dict(unread_notifications=unread)
    return dict(unread_notifications=[])

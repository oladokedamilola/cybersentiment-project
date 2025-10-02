from flask import current_app
from flask_login import current_user
from webapp.models import Notification, NotificationRead, Alert, AlertRead

def unread_counts():
    if current_user.is_authenticated:
        # Get unread notifications
        notification_read_ids = {nr.notification_id for nr in current_user.notification_reads}
        unread_notifications = Notification.query.filter(
            ~Notification.id.in_(notification_read_ids)
        ).order_by(Notification.timestamp.desc()).all()
        
        # Get unread alerts  
        alert_read_ids = {ar.alert_id for ar in current_user.alert_reads if ar.is_read}
        unread_alerts = Alert.query.filter(~Alert.id.in_(alert_read_ids)).all()
        
        return dict(
            unread_notifications=unread_notifications,
            unread_notifications_count=len(unread_notifications),
            unread_alerts_count=len(unread_alerts),
            total_unread_count=len(unread_notifications) + len(unread_alerts)
        )
    return dict(
        unread_notifications=[],
        unread_notifications_count=0,
        unread_alerts_count=0,
        total_unread_count=0
    )
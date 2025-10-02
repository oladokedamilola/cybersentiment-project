from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from webapp.app import db
from webapp.models import Notification, NotificationRead

notifications_bp = Blueprint("notifications", __name__, url_prefix="/notifications")

@notifications_bp.route("/")
@login_required
def index():
    # Get IDs of notifications the user has read
    read_ids = {nr.notification_id for nr in current_user.notification_reads}
    
    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Notifications per page
    
    # Query all notifications with pagination
    notifications_pagination = Notification.query.order_by(
        Notification.timestamp.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)
    
    all_notifications = notifications_pagination.items
    
    # Get unread notifications from the current page
    unread_notifications = [n for n in all_notifications if n.id not in read_ids]
    
    return render_template(
        "notifications.html",
        unread_notifications=unread_notifications,
        all_notifications=all_notifications,
        read_ids=read_ids,
        pagination=notifications_pagination
    )

@notifications_bp.route("/mark_read/<int:notif_id>")
@login_required
def mark_read(notif_id):
    # Check if already read
    existing_read = NotificationRead.query.filter_by(
        notification_id=notif_id,
        user_id=current_user.id
    ).first()
    
    if not existing_read:
        # Mark as read
        notification_read = NotificationRead(
            notification_id=notif_id,
            user_id=current_user.id,
            is_read=True,
            read_at=db.func.now()
        )
        db.session.add(notification_read)
        db.session.commit()
        flash("Notification marked as read.", "success")
    else:
        flash("Notification already read.", "info")
    
    return redirect(request.referrer or url_for('notifications.index'))

@notifications_bp.route("/mark_all_read")
@login_required
def mark_all_read():
    # Get all unread notifications for this user
    read_ids = {nr.notification_id for nr in current_user.notification_reads}
    unread_notifications = Notification.query.filter(
        ~Notification.id.in_(read_ids)
    ).all()
    
    # Mark all as read
    for notification in unread_notifications:
        existing_read = NotificationRead.query.filter_by(
            notification_id=notification.id,
            user_id=current_user.id
        ).first()
        
        if not existing_read:
            notification_read = NotificationRead(
                notification_id=notification.id,
                user_id=current_user.id,
                is_read=True,
                read_at=db.func.now()
            )
            db.session.add(notification_read)
    
    db.session.commit()
    flash("All notifications marked as read.", "success")
    return redirect(url_for('notifications.index'))
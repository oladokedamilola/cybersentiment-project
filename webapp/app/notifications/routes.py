from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_required, current_user
from webapp.app import db
from webapp.models import Notification, NotificationRead

notifications_bp = Blueprint("notifications", __name__, url_prefix="/notifications")

@notifications_bp.route("/")
@login_required
def index():
    # Show all notifications, highlight unread
    read_ids = {nr.notification_id for nr in current_user.read_notifications}
    all_notifications = Notification.query.order_by(Notification.created_at.desc()).all()
    return render_template("notifications/index.html",
                           all_notifications=all_notifications,
                           read_ids=read_ids)

@notifications_bp.route("/read/<int:notification_id>")
@login_required
def mark_read(notification_id):
    notif = Notification.query.get_or_404(notification_id)
    already = NotificationRead.query.filter_by(
        notification_id=notif.id, user_id=current_user.id
    ).first()
    if not already:
        record = NotificationRead(notification_id=notif.id, user_id=current_user.id)
        db.session.add(record)
        db.session.commit()
    flash("Notification marked as read.", "success")
    return redirect(request.referrer or url_for("notifications.index"))

@notifications_bp.route("/read-all")
@login_required
def mark_all_read():
    notifs = Notification.query.all()
    for n in notifs:
        already = NotificationRead.query.filter_by(
            notification_id=n.id, user_id=current_user.id
        ).first()
        if not already:
            record = NotificationRead(notification_id=n.id, user_id=current_user.id)
            db.session.add(record)
    db.session.commit()
    flash("All notifications marked as read.", "success")
    return redirect(url_for("notifications.index"))

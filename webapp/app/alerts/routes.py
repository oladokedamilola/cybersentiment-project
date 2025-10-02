from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from webapp.models import Alert, AlertRead
from webapp.app import db
from sqlalchemy import desc

alerts_bp = Blueprint("alerts", __name__, url_prefix="/alerts")

@alerts_bp.route("/")
@login_required
def index():
    # Allow optional filtering by platform or severity in the future
    platform = request.args.get("platform", "").lower()

    query = Alert.query.order_by(desc(Alert.timestamp))
    if platform in ["twitter", "reddit"]:
        query = query.filter(Alert.platform == platform)

    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Alerts per page
    alerts_pagination = query.paginate(
        page=page, per_page=per_page, error_out=False
    )
    alerts = alerts_pagination.items

    # Get read alert IDs for current user - FIXED: use alert_reads instead of read_alerts
    read_alert_ids = {ar.alert_id for ar in current_user.alert_reads if ar.is_read}

    return render_template(
        "alerts.html",
        alerts=alerts,
        platform=platform,
        pagination=alerts_pagination,
        read_alert_ids=read_alert_ids
    )

@alerts_bp.route("/mark_read/<int:alert_id>")
@login_required
def mark_read(alert_id):
    alert = Alert.query.get_or_404(alert_id)
    
    # Check if already read
    existing_read = AlertRead.query.filter_by(
        alert_id=alert_id,
        user_id=current_user.id
    ).first()
    
    if existing_read:
        if not existing_read.is_read:
            existing_read.is_read = True
            existing_read.read_at = db.func.now()
            db.session.commit()
            flash("Alert marked as read.", "success")
        else:
            flash("Alert already read.", "info")
    else:
        # Mark as read
        alert_read = AlertRead(
            alert_id=alert_id,
            user_id=current_user.id,
            is_read=True,
            read_at=db.func.now()
        )
        db.session.add(alert_read)
        db.session.commit()
        flash("Alert marked as read.", "success")
    
    return redirect(request.referrer or url_for('alerts.index'))

@alerts_bp.route("/mark_all_read")
@login_required
def mark_all_read():
    # Get all unread alerts for this user - FIXED: use alert_reads instead of read_alerts
    read_alert_ids = {ar.alert_id for ar in current_user.alert_reads if ar.is_read}
    unread_alerts = Alert.query.filter(~Alert.id.in_(read_alert_ids)).all()
    
    # Mark all as read
    for alert in unread_alerts:
        existing_read = AlertRead.query.filter_by(
            alert_id=alert.id,
            user_id=current_user.id
        ).first()
        
        if existing_read:
            if not existing_read.is_read:
                existing_read.is_read = True
                existing_read.read_at = db.func.now()
        else:
            alert_read = AlertRead(
                alert_id=alert.id,
                user_id=current_user.id,
                is_read=True,
                read_at=db.func.now()
            )
            db.session.add(alert_read)
    
    db.session.commit()
    flash("All alerts marked as read.", "success")
    return redirect(url_for('alerts.index'))
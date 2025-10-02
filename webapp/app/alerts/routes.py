from flask import Blueprint, render_template, request
from flask_login import login_required
from webapp.models import Alert
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

    alerts = query.all()

    return render_template(
        "alerts.html",
        alerts=alerts,
        platform=platform
    )

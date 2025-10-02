from flask import Blueprint, render_template

main_bp = Blueprint("main", __name__)

@main_bp.route("/")
def index():
    return render_template("index.html")

# Error handlers
@main_bp.app_errorhandler(400)
def bad_request(e):
    return render_template("400.html"), 400

@main_bp.app_errorhandler(401)
def unauthorized(e):
    return render_template("401.html"), 401

@main_bp.app_errorhandler(403)
def forbidden(e):
    return render_template("403.html"), 403

@main_bp.app_errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@main_bp.app_errorhandler(405)
def method_not_allowed(e):
    return render_template("405.html"), 405

@main_bp.app_errorhandler(408)
def request_timeout(e):
    return render_template("408.html"), 408

@main_bp.app_errorhandler(413)
def payload_too_large(e):
    return render_template("413.html"), 413

@main_bp.app_errorhandler(429)
def too_many_requests(e):
    return render_template("429.html"), 429

@main_bp.app_errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500

@main_bp.app_errorhandler(502)
def bad_gateway(e):
    return render_template("502.html"), 502

@main_bp.app_errorhandler(503)
def service_unavailable(e):
    return render_template("503.html"), 503

@main_bp.app_errorhandler(504)
def gateway_timeout(e):
    return render_template("504.html"), 504
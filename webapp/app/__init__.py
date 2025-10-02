from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate

# Extensions
db = SQLAlchemy()
login_manager = LoginManager()
migrate = Migrate()

def create_app():
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static"
    )

    # Load configuration
    from webapp.config import Config
    app.config.from_object(Config)

    # Init extensions
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)

    login_manager.login_view = "auth.login"

    # Import context processor
    from webapp.app.context_processor import unread_notifications
    app.context_processor(unread_notifications)

    # Import and register blueprints (fixed imports)
    from webapp.app.main.routes import main_bp
    from webapp.app.auth.routes import auth_bp
    from webapp.app.dashboard.routes import dashboard_bp
    from webapp.app.alerts.routes import alerts_bp
    from webapp.app.profile.routes import profile_bp
    from webapp.app.posts.routes import posts_bp
    from webapp.app.analytics.routes import analytics_bp
    from webapp.app.notifications.routes import notifications_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(alerts_bp)
    app.register_blueprint(profile_bp)
    app.register_blueprint(posts_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(notifications_bp)

    # Flash category mapping
    @app.template_filter("bootstrap_class")
    def bootstrap_class_filter(category):
        mapping = {
            "message": "info",
            "info": "info",
            "success": "success",
            "warning": "warning",
            "error": "danger",
            "danger": "danger",
        }
        return mapping.get(category, category)

    return app

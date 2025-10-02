from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from webapp.app import db
from webapp.models import Post, Notification, Alert, AlertRead
from webapp.forms import PostSearchForm

posts_bp = Blueprint("posts", __name__, url_prefix="/posts")

@posts_bp.route("/search", methods=["GET", "POST"])
@login_required
def search():
    form = PostSearchForm(request.args)
    query = Post.query

    # Check if any search criteria are provided
    has_search_criteria = any([
        form.keyword.data,
        form.platform.data,
        form.sentiment.data,
        form.date_from.data,
        form.date_to.data
    ])

    if form.validate():
        if form.keyword.data:
            query = query.filter(Post.text.ilike(f"%{form.keyword.data}%"))
        if form.platform.data:
            query = query.filter(Post.platform == form.platform.data)
        if form.sentiment.data:
            # Use predicted_sentiment instead of sentiment
            query = query.filter(Post.predicted_sentiment == form.sentiment.data)
        if form.date_from.data:
            query = query.filter(Post.timestamp >= form.date_from.data)
        if form.date_to.data:
            query = query.filter(Post.timestamp <= form.date_to.data)

    # Limit to 5 posts by default, 50 when searching
    limit = 50 if has_search_criteria else 5
    posts = query.order_by(Post.timestamp.desc()).limit(limit).all()

    return render_template("search.html", form=form, posts=posts, has_search_criteria=has_search_criteria)

@posts_bp.route("/all")
@login_required
def all_posts():
    """Display all posts with pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Posts per page
    
    posts_pagination = Post.query.order_by(Post.timestamp.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    posts = posts_pagination.items

    return render_template("all_posts.html", posts=posts, pagination=posts_pagination)

@posts_bp.route("/<int:post_id>")
@login_required
def post_detail(post_id):
    """Display detailed view of a single post."""
    post = Post.query.get_or_404(post_id)
    
    # Auto-mark any alerts for this post as read
    alerts_for_post = Alert.query.filter_by(post_id=str(post_id)).all()
    for alert in alerts_for_post:
        existing_read = AlertRead.query.filter_by(
            alert_id=alert.id,
            user_id=current_user.id
        ).first()
        
        if not existing_read:
            alert_read = AlertRead(
                alert_id=alert.id,
                user_id=current_user.id,
                is_read=True,
                read_at=db.func.now()
            )
            db.session.add(alert_read)
        elif not existing_read.is_read:
            existing_read.is_read = True
            existing_read.read_at = db.func.now()
    
    db.session.commit()
    
    return render_template("post_detail.html", post=post)

@posts_bp.route("/flag/<int:post_id>")
@login_required
def flag_post(post_id):
    """Manually flag a post as risky and create a global notification."""
    post = Post.query.get_or_404(post_id)
    post.risk_flag = True
    db.session.commit()

    # Create global notification for flagged post
    notif = Notification(
        message=f"⚠️ Post flagged: '{post.text[:50]}...' on {post.platform}"
    )
    db.session.add(notif)
    db.session.commit()

    flash("Post flagged and notification created.", "warning")
    return redirect(request.referrer or url_for('posts.all_posts'))
from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from webapp.app import db
from webapp.models import Post, Notification
from webapp.forms import PostSearchForm

posts_bp = Blueprint("posts", __name__, url_prefix="/posts")

@posts_bp.route("/search", methods=["GET", "POST"])
@login_required
def search():
    form = PostSearchForm(request.args)
    query = Post.query

    if form.validate():
        if form.keyword.data:
            query = query.filter(Post.text.ilike(f"%{form.keyword.data}%"))
        if form.platform.data:
            query = query.filter(Post.platform == form.platform.data)
        if form.sentiment.data:
            query = query.filter(Post.sentiment == form.sentiment.data)
        if form.date_from.data:
            query = query.filter(Post.timestamp >= form.date_from.data)
        if form.date_to.data:
            query = query.filter(Post.timestamp <= form.date_to.data)

    posts = query.order_by(Post.timestamp.desc()).limit(50).all()

    return render_template("posts/search.html", form=form, posts=posts)


@posts_bp.route("/flag/<int:post_id>")
@login_required
def flag_post(post_id):
    """Manually flag a post as risky and create a global notification."""
    post = Post.query.get_or_404(post_id)
    post.flagged = True
    db.session.commit()

    # Create global notification for flagged post
    notif = Notification(
        message=f"⚠️ Post flagged: '{post.text[:50]}...' on {post.platform}"
    )
    db.session.add(notif)
    db.session.commit()

    flash("Post flagged and notification created.", "warning")
    return redirect(url_for("posts.search"))

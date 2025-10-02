from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import login_required, current_user
from webapp.app import db
from webapp.forms import ProfileForm
import os
from flask import current_app, request
from werkzeug.utils import secure_filename
from uuid import uuid4  # To generate unique filenames


profile_bp = Blueprint("profile", __name__, url_prefix="/profile")



@profile_bp.route("/", methods=["GET", "POST"])
@login_required
def index():
    form = ProfileForm(obj=current_user)
    if form.validate_on_submit():
        # Update user fields
        current_user.username = form.username.data
        current_user.email = form.email.data
        current_user.first_name = form.first_name.data
        current_user.last_name = form.last_name.data

        # Handle profile image upload
        if form.profile_image.data:
            image_file = form.profile_image.data
            filename = secure_filename(image_file.filename)

            # Generate a unique filename to avoid collisions
            unique_filename = f"{uuid4().hex}_{filename}"

            # Define the upload path (ensure this folder exists)
            upload_path = os.path.join(current_app.root_path, 'static', 'uploads', unique_filename)

            # Save the file
            image_file.save(upload_path)

            # Update the user's profile_image field with the filename (not full path)
            current_user.profile_image = unique_filename

        # Commit changes
        db.session.commit()
        flash("Profile updated successfully.", "success")
        return redirect(url_for("dashboard.index"))

    return render_template("profile.html", form=form, user=current_user)

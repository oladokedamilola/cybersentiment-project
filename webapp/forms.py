from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Length, Optional


# --- Registration Form ---
class RegistrationForm(FlaskForm):
    username = StringField(
        "Username",
        validators=[DataRequired(), Length(min=3, max=25)]
    )
    email = StringField(
        "Email",
        validators=[DataRequired(), Email()]
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired(), Length(min=6)]
    )
    confirm_password = PasswordField(
        "Confirm Password",
        validators=[DataRequired(), EqualTo("password")]
    )
    submit = SubmitField("Register")


# --- Login Form ---
class LoginForm(FlaskForm):
    email = StringField(
        "Email",
        validators=[DataRequired(), Email()]
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired()]
    )
    submit = SubmitField("Login")


# --- Profile Form ---
from flask_wtf.file import FileField, FileAllowed

class ProfileForm(FlaskForm):
    username = StringField(
        "Username",
        validators=[DataRequired(), Length(min=3, max=25)]
    )
    email = StringField(
        "Email",
        validators=[DataRequired(), Email()]
    )
    first_name = StringField(
        "First Name",
        validators=[Length(max=30)]
    )
    last_name = StringField(
        "Last Name",
        validators=[Length(max=30)]
    )
    profile_image = FileField(
        "Profile Image",
        validators=[FileAllowed(['jpg', 'jpeg', 'png', 'gif'], "Images only!")]
    )
    submit = SubmitField("Update Profile")


from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, DateField, SubmitField
from wtforms.validators import Optional

class PostSearchForm(FlaskForm):
    keyword = StringField("Keyword", validators=[Optional()])
    platform = SelectField(
        "Platform",
        choices=[("", "All"), ("twitter", "Twitter"), ("reddit", "Reddit")],
        default=""
    )
    sentiment = SelectField(
        "Sentiment",
        choices=[("", "All"), ("positive", "Positive"), ("neutral", "Neutral"), ("negative", "Negative")],
        default=""
    )
    date_from = DateField("From Date", validators=[Optional()])
    date_to = DateField("To Date", validators=[Optional()])
    submit = SubmitField("Search")

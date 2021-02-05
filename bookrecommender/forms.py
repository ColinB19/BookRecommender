from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from bookrecommender import db
from bookrecommender.models import User
# https://www.youtube.com/watch?v=UIJKdCIEXUQ


class RegistrationForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=5, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

    #validation
    def validate_username(self, username):
        user  = db.session.query(User).filter_by(username = username.data).first()
        if user:
            raise ValidationError('That username is taken. Please choose another.')
    
    def validate_email(self, email):
        user  = db.session.query(User).filter_by(email= email.data).first()
        if user:
            raise ValidationError('That email is already in use. Please use another.')


class LoginForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

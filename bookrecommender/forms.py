"""
author: Colin Bradley
last updated: 02/17/2021

TODO
----
1. Docstrings, comments, general cleanliness
2. When you try to reset your password, the app SHOULD NOT tell someone whether
or not a useername/password isn't in use.
"""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from bookrecommender import db
from bookrecommender.models import User
from flask_login import current_user
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
        user  = User.query.filter_by(username = username.data).first()
        if user:
            raise ValidationError('That username is taken. Please choose another.')
    
    def validate_email(self, email):
        user  = User.query.filter_by(email= email.data).first()
        if user:
            raise ValidationError('That email is already in use. Please use another.')


class LoginForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

class UpdateAccountForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=5, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    submit = SubmitField('Update')

    #validation
    def validate_username(self, username):
        if username.data != current_user.username:
            user  = User.query.filter_by(username = username.data).first()
            if user:
                raise ValidationError('That username is taken. Please choose another.')
    
    def validate_email(self, email):
        if email.data != current_user.email:
            user  = User.query.filter_by(email= email.data).first()
            if user:
                raise ValidationError('That email is already in use. Please use another.')

class RequestResetForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    submit = SubmitField('Request Password Reset')



class ResetPasswordForm(FlaskForm):
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')

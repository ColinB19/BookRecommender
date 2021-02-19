"""
author: Colin Bradley
last updated: 02/18/2021

TODO
----
1. Docstrings, comments, general cleanliness
2. When you try to reset your password, the app SHOULD NOT tell someone whether
or not a username/password isn't in use.

NOTE: All forms will inherit from FlaskForm.

RESOURCES
---------
1. Flask Forms: https://flask.palletsprojects.com/en/1.1.x/patterns/wtforms/
2. Flask Login: https://flask-login.readthedocs.io/en/latest/
3. Corey Schafer's Flask tutorial: https://www.youtube.com/watch?v=MwZwr5Tvyxo&list=PL-osiE80TeTs4UjLw5MM6OjgkjFeUxCYH
"""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from bookrecommender import db
from bookrecommender.models import User
from flask_login import current_user
# https://www.youtube.com/watch?v=UIJKdCIEXUQ


class RegistrationForm(FlaskForm):
    """
    Creates a registration form for the site. New users are directed here when 
    they try to log in. Also a tab on the home page for this. If users try
    to rate books without an account they will be redirected here as well.

    Users will input desired username, email, and password (with a confirm password).
    The site then verifies this information (no null fields, valid email, passwords match),
    and then pushes the new account to the DB.

    Attributes
    ----------
    username : str
        StringField (Non-Null) puts limits on string length
    email : str
        StringField (Non-Null) checks for valid email
    password : str
        PasswordField (Non-Null) 
    confirm_password : str 
        PasswordField (Non-Null) verifies equaivalency with password
    submit : button
        SubmitField submits data to then execute some code out of routes.py

    Methods
    -------
    validate_username(username=""):
        Validates username as unique.
    validate_email(email=""):
        Validates email as unique.

    """
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=5, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

    # validation
    def validate_username(self, username):
        """
        Takes in a potential username, queries the DB for that username. If it already
        exists the function will raise a validation error.
        """
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError(
                'That username is taken. Please choose another.')

    def validate_email(self, email):
        """
        Takes in a potential email, queries the DB for that email. If it already
        exists the function will raise a validation error.
        """
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError(
                'That email is already in use. Please use another.')


class LoginForm(FlaskForm):
    """
    Provides a login form for users to enter their email and password.

    Attributes
    ----------
    email : str
        StringField (Non-Null) checks for valid email
    password : str
        PasswordField (Non-Null) 
    submit : button
        SubmitField submits data to then execute some code out of routes.py

    """
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')


class UpdateAccountForm(FlaskForm):
    """
    A form on the account page to change a users email/username.

    Attributes
    ----------
    username : str
        StringField (Non-Null) puts limits on string length 
    email : str
        StringField (Non-Null) checks for valid email
    submit : button
        SubmitField submits data to then execute some code out of routes.py

    Methods
    -------
    validate_username(username=""):
        Validates username as unique and makes sure the new username provided is not the
        current username.
    validate_email(email=""):
        Validates email as unique and makes sure the new username provided is not the
        current email.

    """
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=5, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    submit = SubmitField('Update')

    # validation
    def validate_username(self, username):
        """
        Takes in a potential username, checks that it is not
        the users current username, queries the DB for that username.
        If it already exists the function will raise a validation error.
        """
        if username.data != current_user.username:
            user = User.query.filter_by(username=username.data).first()
            if user:
                raise ValidationError(
                    'That username is taken. Please choose another.')

    def validate_email(self, email):
        """
        Takes in a potential email, checks that it is not
        the users current email, queries the DB for that email.
        If it already exists the function will raise a validation error.
        """
        if email.data != current_user.email:
            user = User.query.filter_by(email=email.data).first()
            if user:
                raise ValidationError(
                    'That email is already in use. Please use another.')


class RequestResetForm(FlaskForm):
    """
    A form used when a user tries to reset their password. The form will 
    take in a users email, then the app will send a reset email to that 
    user.

    Attributes
    ----------
    email : str
        StringField (Non-Null) checks for valid email
    submit : button
        SubmitField submits data to then execute some code out of routes.py
    """
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    submit = SubmitField('Request Password Reset')


class ResetPasswordForm(FlaskForm):
    """
    A user wishing to reset their password will be directed here
    by a reset email sent by the app. The user will then be able
    to input a new password and submit it. 

    Attributes
    ----------
    password : str
        PasswordField (Non-Null) 
    confirm_password : str 
        PasswordField (Non-Null) verifies equaivalency with password
    submit : button
        SubmitField submits data to then execute some code out of routes.py
    """
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')

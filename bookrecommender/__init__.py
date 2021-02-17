"""
author: Colin Bradley
last updated: 02/17/2021


TODO
----
1. I'd like to create a config file which keeps all the settings.
"""


import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_mail import Mail

# initialize app and database configuations
app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

LOCAL = False
DEV = True

if DEV:
    app.config['DEBUG'] = True

if LOCAL:
    DB_USERNAME = os.environ.get("DB_USER")
    DB_PASSWORD = os.environ.get("DB_PASS")
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://'+DB_USERNAME+':'+DB_PASSWORD+'@localhost/bookapp'
else: 
    RDS_HOSTNAME = os.environ.get("RDS_HOSTNAME")
    RDS_PORT = os.environ.get("RDS_PORT")
    RDS_DB_NAME = os.environ.get("RDS_DB_NAME")
    RDS_USERNAME = os.environ.get("RDS_USERNAME")
    RDS_PASSWORD = os.environ.get("RDS_PASSWORD")
    app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{RDS_USERNAME}:{RDS_PASSWORD}@{RDS_HOSTNAME}:{RDS_PORT}/{RDS_DB_NAME}"


db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME']= os.environ.get("BR_GMAIL_USER")
app.config['MAIL_PASSWORD']= os.environ.get("BR_GMAIL_PASS")
app.config['MAIL_SUPPRESS_SEND'] = False
app.config['TESTING'] = False
mail = Mail(app)

# imports all of our route pages
from bookrecommender import routes
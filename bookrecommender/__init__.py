"""
author: Colin Bradley
last updated: 02/23/2021


TODO
----
1. I'd like to create a config file which keeps all the settings.
2. With AWS Lambda I can handle many requests at once! Thus I can add
a recommend button so we don't have to batch update. I probably don't want 
to do this for money purposes though.

NOTE: I user a ping service from https://kaffeine.herokuapp.com/ to keep my site active at all times. Turn this off if you
take the site down

RESOURCES
---------
1. Flask Forms: https://flask.palletsprojects.com/en/1.1.x/patterns/wtforms/
2. Flask Login: https://flask-login.readthedocs.io/en/latest/
3. Flask Mail: https://pythonhosted.org/Flask-Mail/
4. Flask bcrypt: https://flask-bcrypt.readthedocs.io/en/latest/
5. Corey Schafer's Flask tutorial: https://www.youtube.com/watch?v=MwZwr5Tvyxo&list=PL-osiE80TeTs4UjLw5MM6OjgkjFeUxCYH
6. SQLAlchemy: https://www.sqlalchemy.org/
7. AWS RDS: https://aws.amazon.com/rds/?nc2=h_ql_prod_db_rds
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

DEV = False # set to true for debug mode and local database

if DEV:
    app.config['DEBUG'] = True
    DB_USERNAME = os.environ.get("DB_USER")
    DB_PASSWORD = os.environ.get("DB_PASS")
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://'+DB_USERNAME+':'+DB_PASSWORD+'@localhost/bookapp'
else: 
    app.config['DEBUG'] = False
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

# imports all of our route pages. This needs
# to be down here to prevent circular imports
from bookrecommender import routes
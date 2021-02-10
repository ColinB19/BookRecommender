import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_mail import Mail

# initialize app and database configuations
app = Flask(__name__)

app.config['SECRET_KEY'] = 'e118b0dd389d8706291ead5d5d1b9932'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db_un = os.environ.get("DB_USER")
db_pw = os.environ.get("DB_PASS")

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://'+db_un+':'+db_pw+'@localhost/bookapp'
app.config['DEBUG'] = True

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
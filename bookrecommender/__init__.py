from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.automap import automap_base
from flask_bcrypt import Bcrypt
from flask_login import LoginManager

# initialize app and database configuations
app = Flask(__name__)

app.config['SECRET_KEY'] = 'e118b0dd389d8706291ead5d5d1b9932'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://colin:1234@localhost/bookapp'
app.config['DEBUG'] = True

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# imports all of our route pages
from bookrecommender import routes
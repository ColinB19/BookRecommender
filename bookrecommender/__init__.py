from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.automap import automap_base

# initialize app and database configuations
app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://colin:1234@localhost/books'
app.config['DEBUG'] = True
db = SQLAlchemy(app)

# creates classes out of all of our tables
Base = automap_base()
Base.prepare(db.engine, reflect = True)

# imports all of our route pages
from bookrecommender import routes
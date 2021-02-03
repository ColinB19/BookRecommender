from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.automap import automap_base

# app = Flask(__name__)

# ENV = 'dev'
# if ENV == 'dev':
#     app.config['DEBUG'] = True
#     # db on home pc
#     app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://colin:1234@localhost/books'
# else:
#     app.config['DEBUG'] = False
#     # db on heroku

# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db = SQLAlchemy(app)

app = Flask(__name__)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://colin:1234@localhost/books'
app.config['DEBUG'] = True

db = SQLAlchemy(app)

Base = automap_base()
Base.prepare(db.engine, reflect = True)

from bookrecommender import routes
"""
Created on Mon Jan 25 17:36:14 2021

@author: colin

"""
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.automap import automap_base

# initialize app and database configuations
app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://colin:1234@localhost/books'
db = SQLAlchemy(app)

# creates classes out of all of our tables
Base = automap_base()
Base.prepare(db.engine, reflect = True)
book = Base.classes.books_meta

# checking a query
queryString = "%king%"
results = db.session.query(book).filter(book.authors.ilike(queryString)).limit(10)
for result in results:
    print(result.authors, result.title, sep = '\t', end = '\n')

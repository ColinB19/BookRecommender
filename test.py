#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:36:14 2021

@author: colin

I cant even query my database. What is going on here? connecting just fine but then nothing is showing up
"""
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.automap import automap_base

app = Flask(__name__)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://colin:1234@localhost/books'

db = SQLAlchemy(app)

# # this line reflects the database meta into the current engine
# db.Model.metadata.reflect(db.engine)

Base = automap_base()
Base.prepare(db.engine, reflect = True)
book = Base.classes.books_meta

queryString = "%king%"

# holy crap I did it.
results = db.session.query(book).filter(book.authors.ilike(queryString)).limit(10)


for result in results:
    print(result.authors, result.title, sep = '\t', end = '\n')


# class book(db.Model):
#     __table__ = db.Model.metadata.tables['books_meta']

#     def __repr__(self):
#         return '<Aothor %r>' % self.authors

#from flask_sqlalchemy.orm import sessionmaker

#book = db.Table('books_meta', db.metadata, autoload = True, autoload_with = db.engine)

# queryString = "%J.K.%"
# #results = db.session.query(book).filter(book.authors.like(queryString))

# for result in results:
#     print(result.authors, result.title, sep = '\t', end = '\n')

# check = Book.query.filter_by(authors = "J.R.R. Tolkien").all()
# for book in check:
#     print(book.title, end = '\n')




# searchResults = Book.query.filter_by(Book.authors.like(fullQuery))
# print(searchResults.title)

# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base

# engine = create_engine('postgresql://colin:1234@localhost/books', convert_unicode=True, echo=False)
# Base = declarative_base()
# Base.metadata.reflect(engine)

# from sqlalchemy.orm import relationship, backref

# class book(Base):
#     __table__ = Base.metadata.tables['books_meta']


# if __name__ == '__main__':
#     from sqlalchemy.orm import scoped_session, sessionmaker, Query
#     db_session = scoped_session(sessionmaker(bind=engine))
#     for item in db_session.query(book.authors, book.title):
#         print(item)
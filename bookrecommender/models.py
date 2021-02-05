"""
Created on Wed Jan 27 20:46:04 2021

@author: colin
"""
from bookrecommender import Base
from bookrecommender import login_manager, db
from flask_login import UserMixin

# figure out how to do cascade on deletes...


@login_manager.user_loader
def load_user(user_id):
    return db.session.query(User).get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.Integer, unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    reviews = db.relationship('UserRating', backref='reader', lazy=True)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"

class UserRating():
    id = db.Column(db.Integer, primary_key=True)
    book_id = db.Column(db.Integer, db.ForeignKey('book.id'), nullable=False)
    site_id=db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"UserRating('{self.site_id}', '{self.book_id}', '{self.rating}')"

class ArchiveRating():
    id = db.Column(db.Integer, primary_key=True)
    book_id = db.Column(db.Integer, db.ForeignKey('book.id'), nullable=False)
    user_id=db.Column(db.Integer, nullable=False)
    rating = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"ArchiveRating('{self.user_id}', '{self.book_id}', '{self.rating}')"


class Book(db.Model):
    book_id = db.Column(db.Integer, primary_key = True)
    goodreads_book_id = db.Column(db.Integer, unique=True)
    best_book_id = db.Column(db.Integer, unique=True)
    work_id = db.Column(db.Integer, unique=True)
    books_count = db.Column(db.Integer)
    isbn = db.Column(db.Text)
    isbn13 = db.Column(db.Real)
    original_publication_year = db.Column(db.Integer)
    original_title = db.Column(db.Text)
    title = db.Column(db.Text)
    laguage_code = db.Column(db.Text)
    average_rating = db.Column(db.Real)
    ratings_count = db.Column(db.Integer)
    work_ratings_count = db.Column(db.Integer)
    work_text_reviews_count = db.Column(db.Integer)
    ratings_1 = db.Column(db.Integer)
    ratings_2 = db.Column(db.Integer)
    ratings_3 = db.Column(db.Integer)
    ratings_4 = db.Column(db.Integer)
    ratings_5 = db.Column(db.Integer)
    image_url = db.Column(db.Text)
    small_image_url = db.Column(db.Text)

    def __repr__(self):
        return f"Book('{self.title}', '{self.authors}')"
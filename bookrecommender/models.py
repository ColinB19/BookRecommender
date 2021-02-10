"""
Created on Wed Jan 27 20:46:04 2021

@author: colin
"""
from bookrecommender import login_manager, db, app
from flask_login import UserMixin
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer



@login_manager.user_loader
def load_user(user_id):
    return db.session.query(User).get(int(user_id))


class User(db.Model, UserMixin):
    #this needs to go to UUID eventually. I just want to get it running for now.
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    reviews = db.relationship('UserRating', backref='reader', lazy=True)

    def get_reset_token(self, expires_sec = 1800):
        s = Serializer(app.config['SECRET_KEY'], expires_sec)
        return s.dumps({'user_id':self.id}).decode('utf-8')

    @staticmethod
    def verify_reset_token(token):
        s = Serializer(app.config['SECRET_KEY'])
        try:
            user_id = s.loads(token)['user_id']
        except:
            return None
        return User.query.get(user_id)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"


class UserRating(db.Model):
    book_id = db.Column(db.Integer, db.ForeignKey('book.book_id', ondelete="CASCADE"), nullable=False, primary_key=True)
    site_id=db.Column(db.Integer, db.ForeignKey('user.id', ondelete="CASCADE"), nullable=False, primary_key=True)
    rating = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"UserRating('{self.site_id}', '{self.book_id}', '{self.rating}')"


class ArchiveRating(db.Model):
    book_id = db.Column(db.Integer, db.ForeignKey('book.book_id', ondelete="CASCADE"), nullable=False, primary_key=True)
    user_id=db.Column(db.Integer, nullable=False, primary_key=True)
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
    isbn13 = db.Column(db.Float)
    authors = db.Column(db.Text)
    original_publication_year = db.Column(db.Float)
    original_title = db.Column(db.Text)
    title = db.Column(db.Text)
    laguage_code = db.Column(db.Text)
    average_rating = db.Column(db.Float)
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

class UserRecommendations(db.Model):
    __tablename__ = "user_recs"

    book_id = db.Column(db.Integer, db.ForeignKey('book.book_id', ondelete="CASCADE"), nullable=False, primary_key=True)
    site_id=db.Column(db.Integer, db.ForeignKey('user.id', ondelete="CASCADE"), nullable=False, primary_key=True)
    score = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f"UserRecommendations('{self.site_id}', '{self.book_id}', '{self.score}')"
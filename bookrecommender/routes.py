from flask import render_template, request
from bookrecommender.model import Book, Rating
from bookrecommender import app, db
from sqlalchemy import or_

results = [{
    'title':'Test Book',
    'authors':'Me',
    'url':'https://images.gr-assets.com/books/1447303603s/2767052.jpg'
}]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods = ["POST"])
def submit():
    if request.method == "POST":
        queryText = request.form["searchQuery"]
        fullQuery = "%" + queryText + "%"
        
        if queryText == '':
            return render_template('index.html', message='Please enter required fields...')
        else:
            try:
                # this doesn't handle mispellings very well
                searchResults = (db.session.query(Book)
                                    .filter((Book.authors.ilike(fullQuery))|(Book.title.ilike(fullQuery)))
                                    .order_by(Book.ratings_count.desc())
                                    .limit(50)
                )
                return render_template("results.html", results=searchResults)
            except:
                print('something went wrong')
                return render_template('index.html', message='Something went wrong...')
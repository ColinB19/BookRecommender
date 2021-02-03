from flask import render_template, request
from bookrecommender.model import Book, Rating
from bookrecommender import app, db
from sqlalchemy import or_

# home page
@app.route('/')
def index():
    return render_template('index.html')

# search bar functionality
@app.route('/submit', methods = ["POST"])
def submit():
    # just make sure we're trying to post
    if request.method == "POST":
        queryText = request.form["searchQuery"]
        fullQuery = "%" + queryText + "%"
        
        # make sure they search something
        if queryText == '':
            return render_template('index.html', message='Please enter required fields...')
        # make sure they are searching something other than one letter
        elif len(queryText) < 2:
            return render_template('index.html', message='Try being a bit more specific...')
        else:
            try:
                # query our database for authors or titles with the input
                # NOTE: this doesn't handle mispellings very well
                searchResults = (db.session.query(Book)
                                    .filter((Book.authors.ilike(fullQuery))|(Book.title.ilike(fullQuery)))
                                    .order_by(Book.ratings_count.desc())
                                    .limit(50)
                )
                return render_template("results.html", results=searchResults)
            except:
                # fault tollerance
                print('404: Error, something went wrong.')
                return render_template('index.html', message='Something went wrong...')
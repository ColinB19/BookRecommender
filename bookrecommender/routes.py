from flask import render_template, request, flash, redirect, url_for
from bookrecommender.model import Book, Rating
from bookrecommender import app, db
from bookrecommender.forms import RegistrationForm, LoginForm

# home page
@app.route('/')
def index():
    return render_template('index.html')

# register
@app.route("/register", methods = ["GET", "POST"])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('index'))
    return render_template('register.html', title = 'Register', form = form)

# register
@app.route("/login", methods = ["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data == 'admin@blog.com' and form.password.data == 'password':
            flash('You have been logged in', 'success')
            return redirect(url_for('index'))
        else:
            flash('Login unsuccessful, please check username and password', 'danger')
    return render_template('login.html', title = 'Login', form = form)

# search bar functionality
@app.route('/submit', methods=["POST"])
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
            # try:
            # query our database for authors or titles with the input
            # NOTE: this doesn't handle mispellings very well
            print(fullQuery, end='\n'*3)
            searchResults = (db.session.query(Book)
                                .filter((Book.authors.ilike(fullQuery)) | (Book.title.ilike(fullQuery)))
                                .order_by(Book.ratings_count.desc())
                                .limit(50)
                                )
            print(searchResults)
            return render_template("results.html", results=searchResults)
            # except:
            #     # fault tollerance
            #     print('404: Error, something went wrong.')
            #     return render_template('index.html', message='404: Error, Something went wrong...')


# @app.route('/rate', methods=['POST'])
# def rate():
#     return render_template('index.html')

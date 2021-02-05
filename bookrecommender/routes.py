from flask import render_template, request, flash, redirect, url_for
from bookrecommender.models import Book, User
from bookrecommender import app, db
from bookrecommender.forms import RegistrationForm, LoginForm


@app.route('/')
@app.route("/home")
def index():
    return render_template('index.html')

@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('index'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data == 'admin@blog.com' and form.password.data == 'password':
            flash('You have been logged in', 'success')
            return redirect(url_for('index'))
        else:
            flash('Login unsuccessful, please check username and password.', 'danger')
    return render_template('login.html', title='Login', form=form)

# search bar functionality


@app.route('/submit', methods=["POST"])
def submit():
    # just make sure we're trying to post
    if request.method == "POST":
        queryText = request.form["searchQuery"]
        fullQuery = "%" + queryText + "%"

        # make sure they search something
        if queryText == '':
            flash('Please enter required fields...', 'danger')
            return redirect(url_for('index'))
        elif len(queryText) < 2:
            flash('Try to be more specific.', 'danger')
            return redirect(url_for('index'))
        else:
            try:
                # query our database for authors or titles with the input
                # NOTE: this doesn't handle mispellings very well
                searchResults = (db.session.query(Book)
                                .filter((Book.authors.ilike(fullQuery)) | (Book.title.ilike(fullQuery)))
                                .order_by(Book.ratings_count.desc())
                                .limit(50)
                                )
                return render_template("results.html", results=searchResults, title = "Search Results")
            except:
                flash('Uh-oh! Something was wrong with your search.', 'danger')
                return redirect(url_for('index'))


# @app.route('/rate', methods=['POST'])
# def rate():
#     return render_template('index.html')

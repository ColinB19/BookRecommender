from flask import render_template, request, flash, redirect, url_for
from bookrecommender import app, db, bcrypt, mail
from bookrecommender.forms import (RegistrationForm, LoginForm,
                                   UpdateAccountForm, RequestResetForm, 
                                   ResetPasswordForm)
from bookrecommender.models import Book, User, UserRating, UserRecommendations
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message


@app.route('/')
@app.route("/home")
def index():
    # Let's just render the most popular books to the front page.
    # I also want to have an update section for the site.
    popbooks = Book.query.order_by(Book.ratings_count.desc()).limit(15)
    return render_template('index.html', popbooks = popbooks)

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        flash("You're already logged in.", 'info')
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username = form.username.data, email = form.email.data, password = hashed_password)
        db.session.add(user)
        db.session.commit()
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        flash("You're already logged in.", 'info')
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email = form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            print(user.id)
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Login unsuccessful, please check email and password.', 'danger')
    return render_template('login.html', title='Login', form=form)



@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('index'))


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
                searchResults = (Book.query
                                    .filter((Book.authors.ilike(fullQuery)) | (Book.title.ilike(fullQuery)))
                                    .order_by(Book.ratings_count.desc())
                                    .limit(50)
                                )
                # if the user has an account, pull their ratings to populate to the page
                if current_user.is_authenticated:
                    user_id = current_user.get_id()
                    userRates = UserRating.query.filter_by(site_id = user_id).all()
                    return render_template("results.html", 
                                            results = searchResults, 
                                            userRates = userRates, 
                                            title = "Search Results")

                return render_template("results.html", 
                                        results=searchResults,   
                                        title = "Search Results")
            except:
                flash('Uh-oh! Something was wrong with your search.', 'danger')
                return redirect(url_for('index'))



@app.route('/account', methods = ['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash("Your account has been updated.", "success")
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    if current_user.is_authenticated:
        user_id = current_user.get_id()
        bookRatings = (db.session.query(Book, UserRating)
                    .filter(Book.book_id == UserRating.book_id)
                    .filter(UserRating.site_id == user_id).order_by(Book.title).all()
                )
        bookRecommendations = (db.session.query(Book, UserRecommendations)
                    .filter(Book.book_id == UserRecommendations.book_id)
                    .filter(UserRecommendations.site_id == user_id).order_by(UserRecommendations.score).all()
                )
    return render_template("account.html", bookratings = bookRatings, bookRecommendations = bookRecommendations, form = form)
    # return render_template('account.html', ratings = books, title='Account')

@app.route('/rate', methods=['POST'])
@login_required
def rate():
    rating = request.form["rating"]
    if current_user.is_authenticated:
        user_id = current_user.get_id()
    
    # this is a super inelegant solution but it works for now!

    book_id = int(rating[1:])
    site_id = user_id
    book_rating = int(rating[0])
    newRating = UserRating(site_id = site_id, book_id = book_id, rating = book_rating)

    check = UserRating.query.filter_by(site_id = site_id, book_id = book_id).first()

    if check is None:
        # just insert the new book rating if they haven't already rated it.
        db.session.add(newRating)
        db.session.commit()
        return redirect(url_for('account'))
    else:
        # delete the old book rating then insert the new one!
        db.session.delete(check)
        db.session.add(newRating)
        db.session.commit()

    return redirect(url_for('account'))

@app.route("/recommend")
@login_required
def recommend():
    pass


def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message('Password Reset Request',
                sender = 'noreply@colinsbookrecommender.com',
                recipients=[user.email]
                )
    msg.body = f'''To reset your password, visit the following link: {url_for('reset_token', token = token, _external=True)}. 
If you did not make this request just ignore this email.
'''
    mail.send(msg)

@app.route('/reset_password', methods=["GET", "POST"])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RequestResetForm()
    user = User.query.filter_by(email = form.email.data).first()
    if user:
        send_reset_email(user)
        flash('If an account associated with this email exists, an email will arive shortly. Be sure to check your spam!', 'info')
        return redirect(url_for('login'))
    if (user is None) and (form.email.data):
        # this is just so we don't let a potential cybercriminal know that there is no user with this email.
        flash('If an account associated with this email exists, an email will arive shortly.', 'info')
        return redirect(url_for('login'))
    return render_template('reset_request.html', title='Reset Password', form=form)


@app.route('/reset_password/<token>', methods=["GET", "POST"])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        flash(f'Your password has been updated.', 'success')
        return redirect(url_for('login'))
    return render_template('reset_token.html', title = 'Reset Passowrd', form = form)
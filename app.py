#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:46:21 2021

@author: colin

I need to create a database that will handle all of the data from goodbooks! use SQLAlchemy for this.

Note: I don't need to be able to add/remove data from the books dataset! All I really need to add is a new user to the ratings
dataset and then retrain my model. Or I could create new users and then retrain the model hourly and just recommend popular books while they wait?
"""
from flask import Flask, render_template, url_for
# from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.config['DEBUG'] = True
    app.run()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:36:14 2021

@author: colin
"""
import recPipeline as pipe
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

check = pipe.Pipeline()

check.preprocess()

for url in check.books.small_image_url:
    if len(url) > 200:
        print('long!')

# train, test, val = check.split_test_train(testTrainFrac = 1, ratingsWithheldFrac= 0.2, ratingsThresh=4)

# model = pipe.Recommender()
# hyper = model.paramSearch(train = train, test = test, num_samples=2, num_threads=20)


#%%

# model.recommend_random(check.ratings, check.books, seed = 1234)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:49:21 2020

@author: colin

https://sites.google.com/a/chromium.org/chromedriver/downloads

install selenium
    pip install selenium
"""


from selenium import webdriver

#%%

PATH = "~/../../media/Data/Code/Book_recommender/Scraper/chromedriver"
driver = webdriver.Chrome(PATH)

driver.get("https://colinb19.github.io")
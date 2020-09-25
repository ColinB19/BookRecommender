#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 18:51:50 2020

@author: colin

This will be a simple script that will just pull all of the book titles, 
authors, and numbers from a single list on goodreads. Check out 
https://www.goodreads.com/list to find a list you want to scrape! I will
add the functionality to scrape the entire list (which requires going from
page to page) at a later date. This is just the first step in building a 
GoodReads webscraper. 

Also  check out https://github.com/OmarEinea/GoodReadsScraper for a
fully thought out scraper! I will be using this as a reference throughout
this process.

We will use BeautifulSoup, requests, and pandas!
    pip install beautifulsoup4
    pip install requests
"""


import requests
from bs4 import BeautifulSoup
import pandas as pd

#setting up scraper
URL = 'https://www.goodreads.com/list/show/50.The_Best_Epic_Fantasy_fiction?page=1'
page = requests.get(URL)

number_of_pages = 35

soup = BeautifulSoup(page.content,'html.parser')
results = soup.find(id = 'all_votes')


# I'm just getting the elements I want here, only the Title, Author, rating,
# and id are necessary.
books = results.find_all('tr')
scraped = []
scraped_collection = [] #this is so I can go back and pull the whole series of individual books later

for el in books :
    title_elem = el.find('a', class_='bookTitle')
    author_elem = el.find('a', class_='authorName')
    av_rating_elem = el.find('span',  class_='minirating')
    url_elem = el.find('div', class_ = 'js-tooltipTrigger tooltipTrigger')['data-resource-id']
    
    # This just checks for unique books. I don't want to recommend series 
    # or bodies of work in the final product, just books.
    if None in (title_elem, author_elem , av_rating_elem, url_elem):
        continue
    if ('Boxed Set' in title_elem.text.strip() 
        or 'Collection' in title_elem.text.strip() 
        or 'Anthology' in title_elem.text.strip()
        or 'Complete Set' in title_elem.text.strip()):
        scraped_collection.append([url_elem,
                title_elem.text.strip(), 
                author_elem.text.strip(), 
                av_rating_elem.text.strip()])

        continue
    scraped.append([url_elem,
                    title_elem.text.strip(), 
                    author_elem.text.strip(), 
                    av_rating_elem.text.strip()])

scraped_clean = []

for el in scraped:
    ID = int(el[0])
    # This just checks if the element is a series or not
    if ('(' in el[1]):
        title = el[1].split('(')[0][:-1]
        series_name = el[1].split('(')[1].split('#')[0][:-2]
        
        # Again, filtering out enitre series. I'm only keeping individual books.
        try:
            volume = int(el[1].split('(')[1].split('#')[1][:-1])
        except ValueError:
            continue
    else:
        title = el[1]
        series_name = 'Stand Alone Novel'
        volume = '0'
    author = el[2]
    rate_string = el[3].split(' ')
    temp = []
    
    # the ratings string has an non-constant structure, this fixes that.
    for word in rate_string:
        try:
            temp.append(float(word.replace(',','')))
        except ValueError:
            pass
    av_rating = temp[0]
    num_of_ratings = int(temp[1])
    scraped_clean.append([ID,
                          title,
                          author,
                          series_name,
                          volume,
                          av_rating,
                          num_of_ratings])
#%%
print(scraped_collection)

#%%

# this is just for ease of exporting, also for visualization if you wanted to add that.
df = pd.DataFrame(scraped_clean,columns = ['id', 
                                           'Title', 
                                           'Author', 
                                           'Series Title', 
                                           'Volume No.',
                                           'Av. Rating', 
                                           'No. of Reviews'])

df.to_csv('Data/gr_book_list_no_'+URL.split('/')[-1]+'.csv',index = False)
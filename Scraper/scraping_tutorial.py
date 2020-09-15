# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:29:42 2020

@author: colin

This script is meant to test BeautifulSoup.

run:
    pip install beautifulsoup4
    pip install requests
"""
# Let's scrape a job search on monster with the requests library! We will be following 
# https://realpython.com/beautiful-soup-web-scraper-python/


import requests
from bs4 import BeautifulSoup
#%%

URL = 'https://www.monster.com/jobs/search/?q=Data-Scientist&where=San-Francisco__2C-CA'
page = requests.get(URL)

#%%
# Let's get a bs4 object and just look at all of the results from the page! If you're unsure
# of this code, check out the html of the page!

soup = BeautifulSoup(page.content,'html.parser')
results = soup.find(id = 'ResultsContainer')
print(results.prettify())
#%%

job_elems = results.find_all('section', class_='card-content')
for el in job_elems:
    print(el,end='\n'*2)
    
#%%

for job_elem in job_elems:
    title_elem = job_elem.find('h2', class_='title')
    company_elem = job_elem.find('div', class_='company')
    location_elem = job_elem.find('div', class_='location')
    if None in (title_elem, company_elem, location_elem):
        continue
    print(title_elem.text.strip())
    print(company_elem.text.strip())
    print(location_elem.text.strip())
    print()

#%%
ml_jobs = results.find_all('h2',
                               string=lambda text: 'machine learning' in text.lower())
print(len(ml_jobs))
#%% 
for ml_job in ml_jobs:
    link = ml_job.find('a')['href']
    print(ml_job.text.strip())
    print(f"Apply here: {link}\n")
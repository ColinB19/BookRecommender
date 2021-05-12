# The Book Spot
------------------

Hello and welcome to The Book Spot! Here you will find all the code I wrote to get my book recommender up, making recommendations, and deployed to an application. If you like machine learning, front-end development, cloud computing or just books then this is the project for you. This project utilizes collaborative based filtering (matrix factorization) to fill latent feature matrices in order to predict ratings on unread books. Please visit my [blog](https://colinb19.github.io/) for more information about this process!

## Motivation


I was reading the last few books of Robert Jordan's Wheel of Time series when it hit me, what will I do after this? Since I am learning data science, the answer came fairly quickly: I need a new series, why don't I write my own program to recommend one to me? I decided to make this project full-stack, meanning I would be in charge of every stage from data gathering to model deployment. I ended up writing a web scraper but using the [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k). I then went to work building a model and deploying it to (Heroku)[https://www.heroku.com/].

## Features


## Code Example


## How to use?

If you want to get up and running with some quick and dirty recommendations, here's how.

1. Head over and download the [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) dataset! You can check out my web scraper if you want to pull your own data.
2. I have everything loaded into a local Postgres database. If you want to use my code with minimal tweaks you'll have to create one and load the data into it. [Here's](https://www.youtube.com/watch?v=qw--VYLpxG4) the tutorial I used. It's a little long but a lot of that time is devoted to teaching you SQL.
3. Install the required packages from requirements.txt in your local env. Many of the packages are for the Heroku deployment. So if you want just enough to make some quick recommendations, run
    pip install numpy pandas sqlalchemy scipy
    

## License



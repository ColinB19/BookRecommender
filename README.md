# [The Book Spot](https://thebookspot.herokuapp.com/)
------------------

Hello and welcome to The Book Spot! Here you will find all the code I wrote to get my book recommender up, making recommendations, and deployed to an application. If you like machine learning, front-end development, cloud computing or just books then this is the project for you. This project utilizes collaborative based filtering (matrix factorization) to fill latent feature matrices in order to predict ratings on unread books. Please visit my [blog](https://colinb19.github.io/) for more information about this process!

If you need any help, have any questions or comments, don't hesitate to contact me! You can find me on [LinkedIn](https://www.linkedin.com/in/colin-bradley-data-motivated/) or [Twitter](https://twitter.com/data_motivated).


## Motivation

I was reading the last few books of Robert Jordan's Wheel of Time series when it hit me, what will I do after this? Since I am learning data science, the answer came fairly quickly: I need a new series, why don't I write my own program to recommend one to me? I decided to make this project full-stack, meanning I would be in charge of every stage from data gathering to model deployment. I ended up writing a web scraper but using the [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k). I then went to work building a model and deploying it to [Heroku](https://www.heroku.com) .


## Features

This is a full-stack project. The code covers: 
- scraping data
- cleaning and visualizing
- creating a model
- deploying the model
- the back and front end to the application. 

Please visit my [portfolio](https://colinb19.github.io/) for a more detailed description of the feautures and the process of building this application. 


## How to use?

If you want to get up and running with some quick and dirty recommendations, here's how.

1. Head over and download the [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) dataset! You can check out my web scraper if you want to pull your own data.
2. I have everything loaded into a local Postgres database. If you want to use my code with minimal tweaks you'll have to create one and load the data into it. [Here's](https://www.youtube.com/watch?v=qw--VYLpxG4) the tutorial I used. It's a little long but a lot of that time is devoted to teaching you SQL. It may also be easier to forgo the database entirely and just change the `read_data` function in `msePipeline.py` to read directly from csv files. This is a functionality I'll be adding later. 
3. Install the required packages from requirements.txt in your local env. Many of the packages are for the Heroku deployment. So if you want just enough to make some quick recommendations, run
    ```pip install numpy pandas sqlalchemy scipy```
4. Now you can get a recommender going! Clone this repo (You'll really only need `msePipeline.py` and `gettingStarted.ipynb` for this) and open up `gettingStarted.ipynb`.
5. Be sure `DEPLOY` is `False`.
6. If you're using a Postgres database be sure to configure your environment variables. The database username, password, and database name are called `DB_USER`, `DB_PASS`, and `DB_NAME` in the code respectively. Also note the names I've given tables in my database: archive_rating (for the goodbook-10k ratings data) and book (for the goodbook-10k book metadata). There are other tables but there is no need for them in this introduction. 
7. Now the easy part! If your database is all set up and your environment variables are set then go ahead and run through `gettingStarted.ipynb` and have fun!
8. Be sure to check out [my blog](https://colinb19.github.io/) to look at even deeper tutorials on my code and the process I went through to get [The Book Spot](https://thebookspot.herokuapp.com/) up and running!


## License

Please see [LICENSE.txt](LICENSE.txt).

"""
author: Colin Bradley
last updated: 03/02/2021

The script reads in the data from an RDS server, creates sparse matrices, performs matrix factorization and then uploads recommendations
to the RDS server to be read by the website.

FIXME: The validation step in gradient descent isnt working in msePipeline.py


TODO
----
1. If the number of users grows too large the gradient descent might be too costly. You need a way to limit the total number of users when performing GD.
"""

import schedule

DEPLOY = False

def recBooks():
    import msePipeline as mp
    # pull in data and format it correctly
    if DEPLOY:
        print('Establishing connection with RDS...')
    else:
        print('Establishing connection with local postgres database...')
    pipeline = mp.MSEPipeline(deploy=DEPLOY, ratingsThresh=4)
    pipeline.preprocess()

    print("Training a model...")
    # train a model and then predict for the site users
    model = mp.MSErec(df = pipeline.archived_ratings)

    # update the gradient based on new feature matrices
    error = model.trainModel()

    print("Performing and commiting recommendations...")
    if DEPLOY:
        # right now this only works for deployed models
        pipeline.user_predictions = model.getPredictions(pipeline.user_predictions)

        # commit these recommendations to the RDS server
        pipeline.commit_recommendations()

    print("Done!")
    return error

def test():
    print('success!')

schedule.every().hour.do(test)

while True:
    schedule.run_pending()

    # mse = recBooks()
    # print(f"The mean square error of this model was {mse[0]}.")
"""
author: Colin Bradley
last updated: 02/23/2021

The script reads in the data from an RDS server, creates sparse matrices, performs matrix factorization and then uploads recommendations
to the RDS server to be read by the website.

FIXME: The validation step in gradient descent isnt working in msePipeline.py


TODO
----
1. If the number of users grows too large the gradient descent might be too costly. You need a way to limit the total number of users when performing GD.
"""
import imports
import json

def rec(event, context):
    # pull in data and format it correctly
    print('Establishing connection with RDS...')
    pipeline = imports.MSEPipeline(deploy=True)
    pipeline.preprocess()
    print("Training a model...")
    # train a model and then predict for the site users
    model = imports.MSErec(df = pipeline.archived_ratings)
    cost = model.trainModel()
    pipeline.user_predictions = model.getPredictions(pipeline.user_predictions)
    print("Commiting recommendations...")
    # commit these recommendations to the RDS server
    pipeline.commit_recommendations()
    print("Done!")
    val = {"model cost":cost[0]}
    
    
    return {
        "body": val,
        "statusCode": 200
    }


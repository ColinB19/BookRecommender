"""
author: Colin Bradley
last updated: 02/17/2021

The script reads in the data from an RDS server, creates sparse matrices, performs matrix factorization and then uploads recommendations
to the RDS server to be read by the website.

FIXME: The validation step in gradient descent isnt working in msePipeline.py


TODO
----
1. If the number of users grows too large the gradient descent might be too costly. You need a way to limit the total number of users when performing GD.
"""

import msePipeline as mp


pipeline = mp.MSEPipeline(deploy=True)
pipeline.preprocess()

model = mp.MSErec(df = pipeline.archived_ratings)
model.trainModel()
pipeline.user_predictions = model.getPredictions(pipeline.user_predictions)

pipeline.commit_recommendations()
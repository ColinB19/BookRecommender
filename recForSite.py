"""
author: Colin Bradley
Date: 02/16/2021

The script reads in the data from an RDS server, creates sparse matrices, performs matrix factorization and then uploads recommendations
to the RDS server to be read by the website.

FIXIT: The validation step in gradient descent isnt working in msePipeline.py
"""

import msePipeline as mp


pipeline = mp.MSEPipeline()
pipeline.preprocess()

model = mp.MSErec(df = pipeline.archived_ratings)
model.trainModel()
model.getPredictions()

pipeline.commit_recommendations(recommendations=model.df)
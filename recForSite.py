import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from scipy.sparse import coo_matrix as cm
import lightfm as lf
import numpy as np

#this needs to be updated for AWS

db_un = os.environ.get("DB_USER")
db_pw = os.environ.get("DB_PASS")

dbString = 'postgresql://'+db_un+':'+db_pw+'@localhost/bookapp'
db = create_engine(dbString)

dbConnect = db.connect()

archived_ratings = pd.read_sql("select * from \"archive_rating\"", dbConnect)
user_ratings = pd.read_sql("select * from \"user_rating\"", dbConnect)

# I want to filter out user's with only a few ratings
grouped_users = user_ratings.groupby('site_id').count()

for idx, group in grouped_users.iterrows():
    if group.rating < 6:
        user_ratings = user_ratings[user_ratings['site_id'] != idx]

# let's map their id's into archive id's
last_archived_user = archived_ratings.user_id.max()
ulist = user_ratings.site_id.unique().tolist()
user_archive_ids = {}
for i in range(len(ulist)):
    arcid = last_archived_user + i + 1
    temp = {ulist[i]:arcid}
    user_archive_ids.update(temp)

user_ratings['user_id'] = user_ratings.site_id.map(user_archive_ids)

# combine the ratings sets, change column names, fix typing
combined = archived_ratings.append(user_ratings[['user_id', 'book_id', 'rating']])
combined.rename(columns={'user_id': 'uid', 'book_id': 'iid'}, inplace=True)
combined.uid = combined.uid - 1
combined.uid = combined.uid.astype(int)
combined.iid = combined.iid.astype(int)

#get sparse matrices
numUsers = combined.uid.max() + 1
numBooks = combined.iid.max() + 1

ratSparse = cm((combined.rating, (combined.uid, combined.iid)),shape=(numUsers, numBooks))

#train model
model = lf.LightFM(loss='warp', no_components = 20)
model.fit(ratSparse, epochs = 30)

# get recommendations
userRecs = pd.DataFrame(columns = ['site_id', 'book_id', 'score'])
items = np.array(combined.iid.unique().tolist())
items.sort()
num_recs = 20
for site_uid, model_uid in user_archive_ids.items():
    previous_ratings = user_ratings[user_ratings['site_id']==site_uid].book_id.tolist()
    previous_ratings_fixed = [x-1 for x in previous_ratings]
    user_items = [x for x in items if x not in previous_ratings_fixed]
    scores = model.predict(np.int(model_uid-1), user_items)
    ordered_scores = scores[np.argsort(-scores)]
    recIds = np.argsort(-scores)
    for i in range(num_recs):
        userRecs.loc[len(userRecs)] = [site_uid, recIds[i], ordered_scores[i]]

userRecs.book_id = userRecs.book_id  + 1

# fresh recommendations.
dbConnect.execute('DELETE FROM user_recs')
for rec in userRecs.iterrows():
    iid = int(rec[1].book_id)
    sid = rec[1].site_id
    score = rec[1].score
    # just delete all the recommendations since we are making fresh ones now
    dbConnect.execute(f'INSERT INTO user_recs(book_id, site_id, score) VALUES ({iid}, {sid}, {score})')



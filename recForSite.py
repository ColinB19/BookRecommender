import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from scipy.sparse import coo_matrix as cm
import lightfm as lf
import numpy as np

dbString = 'postgresql://colin:1234@localhost/bookapp'
db = create_engine(dbString)

dbConnect = db.connect()

archived_ratings = pd.read_sql("select * from \"archive_rating\"", dbConnect)
user_ratings = pd.read_sql("select * from \"user_rating\"", dbConnect)


user_archive_ids = {idx:archived_ratings.user_id.max() + idx for idx in user_ratings.site_id.unique().tolist()}

user_ratings['user_id'] = user_ratings.site_id.map(user_archive_ids)
combined = archived_ratings.drop('id', axis = 1).append(user_ratings[['user_id', 'book_id', 'rating']])
combined.rename(columns={'user_id': 'uid', 'book_id': 'iid'}, inplace=True)
combined.uid = combined.uid - 1

numUsers = combined.uid.max() + 1
numBooks = combined.iid.max() + 1

ratSparse = cm((combined.rating, (combined.uid, combined.iid)),shape=(numUsers, numBooks))

model = lf.LightFM(loss='warp', no_components = 20)
model.fit(ratSparse, epochs = 30)

userRecs = pd.DataFrame(columns = ['uid', 'iid', 'score'])
items = np.array(combined.iid.unique().tolist())
items.sort()
num_recs = 20
for uid in user_archive_ids.values():
    previous_ratings = user_ratings[user_ratings['site_id']!=uid].book_id.tolist()
    previous_ratings_fixed = [x-1 for x in previous_ratings]
    user_items = [x for x in items if x not in previous_ratings_fixed]
    scores = model.predict(np.int(uid-1), user_items)
    ordered_scores = scores[np.argsort(-scores)]
    recIds = np.argsort(-scores)
    for i in range(num_recs):
        userRecs.loc[len(userRecs)] = [uid, recIds[i], ordered_scores[i]]

userRecs.iid = userRecs.iid  + 1
userRecs['site_id'] = userRecs.uid.map({val:key for key, val in user_archive_ids.items()})
userRecs = userRecs[['iid', 'site_id', 'score']].rename(columns = {'iid':'book_id'})

# fresh recommendations.
dbConnect.execute('DELETE FROM user_recs')
dbConnect.execute('ALTER SEQUENCE user_recs_id_seq RESTART')
for rec in userRecs.iterrows():
    iid = int(rec[1].book_id)
    sid = rec[1].site_id
    score = rec[1].score
    # just delete all the recommendations since we are making fresh ones now
    dbConnect.execute(f'INSERT INTO user_recs(book_id, site_id, score) VALUES ({iid}, {sid}, {score})')



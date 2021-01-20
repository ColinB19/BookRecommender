#standard stuff
import numpy as np
import pandas as pd

# model imports
import random
from scipy.sparse import coo_matrix as cm
import lightfm as lf

# table formatting
from IPython.display import display, HTML

class Pipeline():

    def __init__(self):
        books = pd.DataFrame()
        ratings = pd.DataFrame()
        book_tags = pd.DataFrame()
        tags = pd.DataFrame()
        to_read = pd.DataFrame()

    def preprocess(self):
        '''
        This function is just a clean way to call all preprocessing steps. These steps
        include reading in the goodbooks-10k data, fixing some book id's and more.

        input: 
            self - just class object

        output: 
            null - no need to return anything
        '''
        self.read_data()
        self.fix_ids()

    def get_model(self, epochs=20, num_threads=2, loss='warp'):
        '''
        This is just a clean way to get the model. Creates interaction sparse matrices
        and then trains a model on them. Note: right now this is purely collaborative filtering,
        LightFM supports Hybrid filtering which we will incorporate in the next version.

        input: 
            self - class object.
            epochs - number of epochs for gradient descent.
            num_threads - number of physical cores to parallelize over.
            loss - loss function type. See LightFM Documentation. 

        output: 
            model - our trained model.
        '''
        interactions = self.get_interactions()
        # user_meta, item_meta = self.get_meta()
        model = self.fit_model(interactions, epochs, num_threads, loss)

        return model
    
    
    def read_data(self):
        '''
        This function just reads in the goodbooks-10k data. Later should have the 
        functionality to read in scraped data.

        input: 
            self - class object

        output: 
            null - no need to return anything
        '''
        self.books = pd.read_csv("goodbooks-10k/books.csv")
        self.ratings = pd.read_csv("goodbooks-10k/ratings.csv")

        # may not use these at first.
        self.book_tags = pd.read_csv("goodbooks-10k/book_tags.csv")
        self.tags = pd.read_csv("goodbooks-10k/tags.csv")
        self.to_read = pd.read_csv("goodbooks-10k/to_read.csv")

    def fix_ids(self):
        '''
        This function sets the bookand user id's to start at zero. It also changes the name of 
        headers from user_id and book_id to uid and iid.

        input: 
            self - class object

        output: 
            null - no need to return anything
        '''
        # just changing to standard feature names
        self.ratings.rename(
            columns={'user_id': 'uid', 'book_id': 'iid'}, inplace=True)
        self.books.rename(columns={'book_id': 'iid'}, inplace=True)
        self.to_read.rename(
            columns={'user_id': 'uid', 'book_id': 'iid'}, inplace=True)

        # starting user and book indices from 0
        self.ratings.uid = self.ratings.uid - 1
        self.ratings.iid = self.ratings.iid - 1
        self.books.iid = self.books.iid - 1
        self.to_read.iid = self.to_read.iid - 1

        # this makes all the tags indexed by book_id instead of goodreads_book_id
        temp_books = self.books.set_index('goodreads_book_id')
        idMAP = temp_books['iid'].to_dict()
        self.book_tags['iid'] = self.book_tags.goodreads_book_id.map(idMAP)
        self.book_tags.drop('goodreads_book_id', inplace=True,axis = 1)

    def get_interactions(self):
        '''
        This function gets the interactions sparse matrix using a scipy sparse coo_matrix.

        inputs:
            self - class object

        outputs:
            ratSparse - sparse interactions matrix. 
        '''
        numUsers = self.ratings.uid.max()+1
        numBooks = self.ratings.iid.max()+1

        ratSparse = cm((self.ratings.rating, (self.ratings.uid, self.ratings.iid)),
                       shape=(numUsers, numBooks))
        return ratSparse

#     def get_meta(self):
#         numUsers = self.ratings.uid.max()+1
#         numBooks = self.ratings.iid.max()+1
#         numUserFeatures = self.to_read.book_id.max+1
#         numItemFeatures = self.tags.book_id.max+1

    def fit_model(self, ratSparse, epochs, num_threads, loss):
        '''
        This gets a LightFM model and trains it on the sparse matrix.

        inputs:
            self - class object.
            ratSparse - sparse interactions matrix.
            epochs - number of epochs for gradient descent.
            num_threads - number of physical cores to parallelize over.
            loss - loss function type. See LightFM Documentation. 

        outputs:
            model - our trained model.

        '''

        model = lf.LightFM(loss=loss)
        model.fit(ratSparse, epochs=epochs, num_threads=num_threads)

        return model

    def recommend_random(self, seed, model):
        '''
        This function will pick a random user out of the list of known users and print out their top 
        10 recommendations! It will also print their top 10 rated items for comparison.

        inputs:
            self - class object.
            seed - random seed for reproducibility. 

        outputs:
            null - no need to output anything right now.

        '''

        random.seed(seed)
        user = random.choices(self.ratings.uid.unique().tolist())[0]

        # now let's predict them on our trained model
        itemList = np.array(self.ratings.iid.unique().tolist())
        itemList.sort()

        knownRatings = pd.merge(self.ratings.query('uid == @user'),
                                self.books[['iid', 'title', 'authors']], on='iid', how='left')

        score = model.predict(user, itemList)
        suggested = self.books.loc[np.argsort(-score)][['title', 'authors']]

        print(color.BOLD + 'User {} known items: '.format(user) + color.END, end='\n')
        display(HTML(knownRatings[['title', 'authors', 'rating']]
                     .sort_values(by='rating', ascending=False).iloc[:10].to_html()))
        print(color.BOLD + 'Top 10 suggested items:' + color.END, end='\n')
        display(HTML(suggested[:10].to_html()))
        
        
        
        
class color:
    '''
    This is for printing purposes.
    '''
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'        
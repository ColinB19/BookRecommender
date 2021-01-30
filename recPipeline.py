#standard stuff
import numpy as np
import pandas as pd

# model imports
import random
from scipy.sparse import coo_matrix as cm
import lightfm as lf
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

# table formatting
from IPython.display import display, HTML

# for progress bars
from tqdm import tqdm

class Pipeline():

    def __init__(self):
        self.books = pd.DataFrame()
        self.ratings = pd.DataFrame()
        self.book_tags = pd.DataFrame()
        self.tags = pd.DataFrame()
        self.to_read = pd.DataFrame()

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

    def get_interactions(self, ratingsThresh = 0):
        '''
        This function gets all of the interactions sparse matrix using a scipy sparse coo_matrix.

        Parameters
        ----------
        ratingsThresh : int, optional
            If you only want to include ratings over a certain number. The default is 0.

        Returns
        -------
        ratSparse : TYPE
            DESCRIPTION.

        '''
        
        # only look at ratings above ratingsThresh
        self.ratings = self.ratings.query('rating >= @ratingsThresh')
        
        numUsers = self.ratings.uid.max()+1
        numBooks = self.ratings.iid.max()+1

        ratSparse = cm((self.ratings.rating, (self.ratings.uid, self.ratings.iid)),
                       shape=(numUsers, numBooks))
        return ratSparse

    def split_test_train(self,
                         testTrainFrac = 0.5, 
                         ratingsWithheldFrac = 0.4,
                         testValFrac = 0.5,
                         ratingsThresh = 0):
        '''
        This function takes in the entire ratings dataset and splits the interaction matrices into a training, testing and validation set.
        First the function splits off a portion of reviews that are then split up into test and validation sets. This function is different
        from scikit-learn in that it preserves all user-ids in the training set to avoid a cold start problem when testing your algorithm. 

        Parameters
        ----------
        testTrainFrac : flaot, optional
            What percentage of the users would you like to test/validate on. The default is 0.5.
        ratingsWithheldFrac : float, optional
            What percentage of the ratings of the test/val users would you like to withhold. The default is 0.4.
        testValFrac : float, optional
            Fraction to split test into test/validation. The default is 0.5.
        ratingsThresh : int, optional
            If you only want to include ratings over a certain number. The default is 0.

        Returns
        -------
        sparseTrain : scipy coo matrix
            A sparse matrix of the training user-item interactions. 
        sparseTest : scipy coo matrix
            A sparse matrix of the testing user-item interactions. 
        sparseVal : scipy coo matrix
            A sparse matrix of the validation user-item interactions. 
        '''
        import random
        random.seed(1001)


        # only look at ratings above ratingsThresh
        self.ratings = self.ratings.query('rating >= @ratingsThresh')
        
        # let's get the list of users for the test set
        uids = self.ratings.uid.unique().tolist()
        test_uids = random.sample(uids, k = int(len(uids)*testTrainFrac))

        # get all data for test users and put the rest in training
        test_users = self.ratings.query('uid == @test_uids')
        train = self.ratings.query('uid != @test_uids')

        # only consider 40% of test users ratings and put the rest in the train set.
        # This prevents the cold start problem on the test set. We will later incorporate 
        # new users. 
        test = test_users.groupby('uid').sample(frac = ratingsWithheldFrac, random_state = 1)
        validation = test.groupby('uid').sample(frac = testValFrac, random_state = 888)
        train = train.append(test_users.drop(test.index), ignore_index = True)
        test.drop(validation.index,inplace = True)

        # get these into sparse matrices
        trainNumUsers = train.uid.max()+1
        trainNumBooks = train.iid.max()+1

        sparseTrain = cm((train.rating, (train.uid, train.iid)),
                             shape=(trainNumUsers, trainNumBooks))

        testNumUsers = test.uid.max()+1
        testNumBooks = test.iid.max()+1

        sparseTest = cm((test.rating, (test.uid, test.iid)),
                            shape=(testNumUsers, testNumBooks))

        valNumUsers = validation.uid.max()+1
        valNumBooks = validation.iid.max()+1

        sparseVal = cm((validation.rating, (validation.uid, validation.iid)),
                               shape=(valNumUsers, valNumBooks))
        
        return sparseTrain, sparseTest, sparseVal
        
#     def get_meta(self):
#         numUsers = self.ratings.uid.max()+1
#         numBooks = self.ratings.iid.max()+1
#         numUserFeatures = self.to_read.book_id.max+1
#         numItemFeatures = self.tags.book_id.max+1


class Recommender():
    
    def __init__(self):
        self.model = lf.LightFM(loss = 'warp')
    
    def fit(self,
            interactions,
            no_components = 10,
            learning_schedule = 'adagrad',
            loss='warp',
            learning_rate = 0.05,
            item_alpha = 0.0,
            user_alpha = 0.0,
            max_sampled = 10,
            epochs=20, 
            num_threads=2):
        '''
        this function trains our SVD model. 

        Parameters
        ----------
        interactions : coo_sparse matrix
            sparse user-items interaction training matrix
        rest : see LightFM documentation.
               https://making.lyst.com/lightfm/docs/lightfm.html

        Returns
        -------
        None.

        '''
        # user_meta, item_meta = self.get_meta()
        self.model = lf.LightFM(no_components = no_components,
                                learning_schedule = learning_schedule,
                                loss=loss,
                                learning_rate = learning_rate,
                                item_alpha = item_alpha,
                                user_alpha = user_alpha,
                                max_sampled = max_sampled)
        self.model.fit(interactions, epochs=epochs, num_threads=num_threads)

    def paramSearch(self, train, test, num_samples = 10, num_threads = 2):
        '''
        This function trains models on various sets of random hyperparameters
        and returns a dataframe with scores. 
        
        Parameters
        ----------
        interactions : coo_sparse matrix
            sparse user-items interaction training matrix
        num_samples : int
            How many random samples do you want to pull? The default is 10
        num_threads : int
            number of threads to parallelize over. the default is 2.

        Returns
        -------
        hyperparameters with metric scores.

        '''
        hyperparams = pd.DataFrame()
        print('Searching for optimal parameters...')
        for i in tqdm(range(num_samples)):
            # get a random sample of hyperparameters
            params = sample_hyperparameters()

            # train a model using those hyperparameters
            self.fit(train, **params, num_threads = num_threads)

            prec, rec, auc = self.score(test = test, trian = train, num_threads = num_threads)

            params['precision_mean'] = prec.mean()
            params['recall_mean'] = rec.mean()
            params['auc_mean'] = auc.mean()

            hyperparams = hyperparams.append(params, ignore_index=True).sort_values(by = 'precision_mean')
        return hyperparams
    
    def score(self, test, train, num_threads = 2):
        prec = precision_at_k(self.model,
                              test_interactions = test,
                              train_interactions = train,
                              num_threads=num_threads)
        rec = recall_at_k(self.model,
                          test_interactions = test,
                          train_interactions = train,
                          num_threads=num_threads)
        auc = auc_score(self.model,
                        test_interactions = test,
                        train_interactions = train,
                        num_threads=num_threads)
        return prec, rec, auc
        
    
    def recommend_random(self, ratings, books, seed):
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
        user = random.choices(ratings.uid.unique().tolist())[0]

        # now let's predict them on our trained model
        itemList = np.array(ratings.iid.unique().tolist())
        itemList.sort()

        knownRatings = pd.merge(ratings.query('uid == @user'),
                                books[['iid', 'title', 'authors']], on='iid', how='left')

        score = self.model.predict(user, itemList)
        suggested = books.loc[np.argsort(-score)][['title', 'authors']]

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

def sample_hyperparameters():
    """
    Yield possible hyperparameter choices.
    """

    return {
        "no_components": np.random.randint(5, 64),
        "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
        "loss": np.random.choice(["bpr", "warp", "warp-kos"]),
        "learning_rate": np.random.exponential(0.05),
        "item_alpha": np.random.exponential(1e-8),
        "user_alpha": np.random.exponential(1e-8),
        "max_sampled": np.random.randint(5, 15),
        "epochs": np.random.randint(5, 50),
        }      
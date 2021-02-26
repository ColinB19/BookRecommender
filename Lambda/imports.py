"""
author: Colin Bradley
last updated: 02/23/2021

FIXME: Val step in SGD not working.

TODO
----
1. Docstrings, comments, general cleanliness
2. Still improving hyperparam optimization.
3. Parallelize SGD

"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from scipy import sparse


class MSEPipeline():
    '''
    This class is designed to pull book/user data and perform all the preprocessing necessary to trian
    a model. It will pull data from an RDS server, clean up column names and some labels, and split
    up test, train and validation sets.
    '''

    def __init__(self, deploy=False):
        self.archived_ratings = pd.DataFrame()
        self.user_ratings = pd.DataFrame()
        self.user_archive_ids = {}
        self.deploy = deploy

    def preprocess(self):
        '''
        This function is just a clean way to call all preprocessing steps. These steps
        include reading in the goodbooks-10k data, fixing some book id's and more.
        '''
        self.read_data()
        if self.deploy:  # only add site users to df if we are deploying
            self.remove_new_users()
            self.add_users_to_archive()
        self.fix_ids()

    def read_data(self):
        '''
        Reads the data from an Amazon RDS database. 

        TODO
        ----
        1. Add functionality to pull from local database.
        '''

        # the following lines set up SQL alchemy to grab data from my RDS database. You may need to adjust
        # this to fit yours, or to do something locally.
        RDS_HOSTNAME = os.environ.get("RDS_HOSTNAME")
        RDS_PORT = os.environ.get("RDS_PORT")
        RDS_DB_NAME = os.environ.get("RDS_DB_NAME")
        RDS_USERNAME = os.environ.get("RDS_USERNAME")
        RDS_PASSWORD = os.environ.get("RDS_PASSWORD")
        engine = create_engine(
            f"postgresql://{RDS_USERNAME}:{RDS_PASSWORD}@{RDS_HOSTNAME}:{RDS_PORT}/{RDS_DB_NAME}")
        with engine.connect() as connection:
            self.archived_ratings = pd.read_sql_table(
                'archive_rating', connection)
            if self.deploy:  # only query site users if we are deploying
                self.user_ratings = pd.read_sql_table(
                    'user_rating', connection)

    def remove_new_users(self):
        # check if users have more than 5 reviews
        grouped_users = self.user_ratings.groupby('site_id').count()
        # delete all users who don't
        for idx, group in grouped_users.iterrows():
            if group.rating < 6:
                self.user_ratings = self.user_ratings[self.user_ratings['site_id'] != idx]

    def add_users_to_archive(self):
        last_archived_user = self.archived_ratings.user_id.max()
        ulist = self.user_ratings.site_id.unique().tolist()
        for i in range(len(ulist)):
            arcid = last_archived_user + i + 1
            temp = {ulist[i]: arcid}
            self.user_archive_ids.update(temp)

        self.user_ratings['user_id'] = self.user_ratings.site_id.map(
            self.user_archive_ids)

        # combine the ratings sets, change column names, fix typing
        self.archived_ratings = self.archived_ratings.append(
            self.user_ratings[['user_id', 'book_id', 'rating']])

        # now we need to create a dataframe that will have every book for each user.
        # We will use this in the prediction step, since our predict() method only
        # works on book_id's that exist in the dataframe.
        bookList = self.archived_ratings.book_id.unique().tolist()
        userList = list(self.user_archive_ids.values())
        dfList = []
        for user in userList:
            for book in bookList:
                dfList.append([user, book, np.nan])
        self.user_predictions = pd.DataFrame(
            dfList, columns=['user_id', 'book_id', 'prediction'])

    def commit_recommendations(self):
        try:
            self.user_predictions.iid = self.user_predictions.iid + 1
            self.user_predictions.uid = self.user_predictions.uid + 1
            recs = self.user_predictions[self.user_predictions.uid.isin(
                list(self.user_archive_ids.values()))]
            mapping = {x: y for y, x in self.user_archive_ids.items()}

            RDS_HOSTNAME = os.environ.get("RDS_HOSTNAME")
            RDS_PORT = os.environ.get("RDS_PORT")
            RDS_DB_NAME = os.environ.get("RDS_DB_NAME")
            RDS_USERNAME = os.environ.get("RDS_USERNAME")
            RDS_PASSWORD = os.environ.get("RDS_PASSWORD")

            engine = create_engine(
                f"postgresql://{RDS_USERNAME}:{RDS_PASSWORD}@{RDS_HOSTNAME}:{RDS_PORT}/{RDS_DB_NAME}")
            with engine.connect() as connection:
                connection.execute('DELETE FROM user_recs')
                for rec in recs.iterrows():
                    iid = int(rec[1].iid)
                    sid = mapping[rec[1].uid]
                    score = rec[1].prediction
                    user_books = self.user_ratings[self.user_ratings.site_id == sid].book_id.tolist(
                    )
                    if iid not in user_books:
                        connection.execute(
                            f"INSERT INTO user_recs(book_id, site_id, score) VALUES ({iid}, {sid}, {score})")
        except:
            # FIXME This is bad practice. You should handle specific exceptions.
            print(
                "Something went wrong, maybe you called this method without providing predictions.")

    def fix_ids(self):
        '''
        This function sets the bookand user id's to start at zero. It also changes the name of 
        headers from user_id and book_id to uid and iid.

        TODO
        ----
        1. Include the book/user metadata and clean that up as well.
        '''
        # just changing to standard feature names
        self.archived_ratings.rename(
            columns={'user_id': 'uid', 'book_id': 'iid'}, inplace=True)

        # starting user and book indices from 0
        self.archived_ratings.uid = self.archived_ratings.uid - 1
        self.archived_ratings.iid = self.archived_ratings.iid - 1

        # do the same for user_predictions if we are deploying.
        if self.deploy:
            self.user_predictions.rename(
                columns={'user_id': 'uid', 'book_id': 'iid'}, inplace=True)
            self.user_predictions.uid = self.user_predictions.uid - 1
            self.user_predictions.iid = self.user_predictions.iid - 1


def create_sparse_matrix(df, rows, cols, column_name="rating"):
    ''' 
    Creates a scipy sparse matrix
    Parameters
    ----------
    df : pandas DataFrame
        The data that will be made a sparse matrix
    rows : int
        number of rows in the matrix
    columns : int
        number of columns in the matrix
    column_name : 

    Returns
    -------

    TODO
    ----

    '''
    return sparse.csc_matrix((df[column_name].values, (df['uid'].values, df['iid'].values)), shape=(rows, cols))


def create_embeddings(n, K, gamma=7):
    ''' 

    Parameters
    ----------

    Returns
    -------

    TODO
    ----

    '''
    return gamma*np.random.rand(n, K) / K


def predict(df, user_features, item_features):
    ''' 
    This function performs the element wise prediction of each item for each user. It avoids building the 
    approximated utility matrix in order to save space

    Parameters
    ----------
    df : pandas DataFrame
        This is the pandas dataframe of the data predictions are to be made on.
    user_features : numpy array
        The user feature embeddings.
    item_features : numpy Array
        The item feature embeddings.

    Returns
    -------
    df : pandas DataFrame
        The same dataframe as inputted but with a new/updated predictions column. 

    '''
    df['prediction'] = np.sum(np.multiply(
        item_features[df['iid']], user_features[df['uid']]), axis=1)
    return df


def meanSquareError(df, user_features, item_features):
    ''' 
    Computes the mean square error on the predictions. 

    Parameters
    ----------
    df : pandas DataFrame
        This is the pandas dataframe of the data predictions are to be made on.
    user_features : numpy array
        The user feature embeddings.
    item_features : numpy Array
        The item feature embeddings.

    Returns
    -------
    mse : float
        The mean square error for the given embedding matrices. 

    '''
    # we need to actually make predictions then convert those into a sparse matrix
    utility = create_sparse_matrix(
        df, user_features.shape[0], item_features.shape[0])
    temp = predict(df=df, user_features=user_features,
                   item_features=item_features)
    prediction = create_sparse_matrix(
        temp, user_features.shape[0], item_features.shape[0], 'prediction')

    # now let's get an error matrix then return the MSE.
    error = utility-prediction
    mse = (1/len(df))*np.sum(error.power(2))
    return mse


def gradient_reg(df, utility, user_features, item_features, lmbda_a, lmbda_b):
    ''' 
    Computes the regularized gradient of the mean square error. Returns the gradient
    in the 'directions' of both embedded matrices.

    Parameters
    ----------
    df : pandas DataFrame
        This is the pandas dataframe of the data predictions are to be made on.
    utility : scipy sparse matrix
        The sparse utility matrix of all of the ratings.
    user_features : numpy array
        The user feature embeddings.
    item_features : numpy Array
        The item feature embeddings.
    lmbda_a, lmbda_b : float
        These parameters are the regularization coefficients. 

    Returns
    -------
    grad_user : numpy array
        gradient of the MSE, partial derivative w.r.t. the user 
    grad_item : numpy array
        gradient of the MSE, partial derivative w.r.t. the item 

    '''
    # we need to actually make predictions then convert those into a sparse matrix
    temp = predict(df=df, user_features=user_features,
                   item_features=item_features)
    prediction = sparse.csc_matrix((temp.prediction.values, (temp.uid.values, temp.iid.values)),
                                   shape=(user_features.shape[0], item_features.shape[0]))
    # now let's get an error matrix
    error = utility-prediction

    # we can now compute the gradient
    # we will compute each 'direction' separately and return them separately
    grad_user = (-2/df.shape[0]) * \
        (error*item_features) + 2*lmbda_a*user_features
    grad_item = (-2/df.shape[0])*((error.T) *
                                  user_features) + 2*lmbda_b*item_features
    return grad_user, grad_item


def gradient_descent(df,
                     utility,
                     user_features,
                     item_features,
                     val=None,
                     lmbda_a=0.002,
                     lmbda_b=0.002,
                     epochs=200,
                     learning_rate=0.05,
                     beta=0.9,
                     updates=True,
                     dfError=None
                    ):
    ''' 
    Performs gradient descent to find the optimal embedded matrices. A momentum term
    is added to arrive at the minimum sooner. This function will iterate a number of times
    specified by the user. It will update the user every 50 epochs on how the cost function 
    looks. Finally it will return the new embedded matrices and the final cost values.

    Parameters
    ----------
    df : pandas DataFrame
        This is the pandas dataframe of the data predictions are to be made on.
    utility : scipy sparse matrix
        The sparse utility matrix of all of the ratings.
    user_features : numpy array
        The user feature embeddings.
    item_features : numpy Array
        The item feature embeddings.
    val : pandas DataFrame DEFAULT=None
        The validation set to check the algorithm against.
    lmbda_a, lmbda_b : float, DEFAULT=0.002 for both
        These parameters are the regularization coefficients. 
    epochs : int, DEFAULT=200
        The number of iterations on which to perform GD
    learning_rate : float, DEFAULT=0.05
        The learning rate for GD.
    beta : float, DEFAULT=0.9
        The momentum coefficient.
    updates: bool, DEFAULT=True
        The option to print periodic updates of the MSE as the algorithm runs.
        Updates will print every epoch with the MSE of the set. It will give
        the MSE of the validation set if provided.

    Returns
    -------
    user_features : numpy array
        The optimized user feature embeddings.
    item_features : numpy Array
        The optimized item feature embeddings.
    mse_train : float
        The final MSE of the training set
    mse_val : float, OPTIONAL
        the final MSE of the validation set


    '''

    # get the initial gradient term so we can perform the first
    # round of GD. Needed for momentum terms
    grad_user, grad_item = gradient_reg(df=df,
                                        utility=utility,
                                        user_features=user_features,
                                        item_features=item_features,
                                        lmbda_a=lmbda_a,
                                        lmbda_b=lmbda_b)
    v_user = grad_user
    v_item = grad_item
    for i in range(epochs):
        # update the gradient based on new feature matrices
        grad_user, grad_item = gradient_reg(df=df,
                                            utility=utility,
                                            user_features=user_features,
                                            item_features=item_features,
                                            lmbda_a=lmbda_a,
                                            lmbda_b=lmbda_b)

        # compute our update matrices
        v_user = beta*v_user + (1-beta)*grad_user
        v_item = beta*v_item + (1-beta)*grad_item

        # update the embedded matrices
#         user_features = user_features - learning_rate*v_user
#         item_features = item_features - learning_rate*v_item
        
        user_features = user_features - learning_rate*grad_user
        item_features = item_features - learning_rate*grad_item

        # just print out values every so often to see what is happening 
        # with the algo.
        if(not (i+1) % 50) and (updates):
            print("\niteration", i+1, ":")
            print("train mse:",  meanSquareError(
                df, user_features, item_features))
            if val is not None:
                print("validation mse:",  meanSquareError(
                    val, user_features, item_features))
        if dfError is not None:
            dfError = dfError.append([[i, meanSquareError(df, user_features, item_features)]])

    # compute the final MSE
    mse_train = meanSquareError(df, user_features, item_features)

    # here we just check if the validation set is passed in so we can return the final cost of that as well if needed.
    if val is not None:
        mse_val = meanSquareError(val, user_features, item_features)
        if dfError is not None:
                return (user_features, item_features, mse_train, mse_val, dfError)
        return (user_features, item_features, mse_train, mse_val)
    if dfError is not None: 
        return (user_features, item_features, mse_train, dfError)
    return (user_features, item_features, mse_train)


class MSErec():
    '''
    This class will perform all ML processes to predict books for users of our app. It will create 
    user/item matrices, perform gradient descent (with momentum), and output predictions!
    '''

    def __init__(self, df, test=None, validation=None):
        ''' 
        Parameters
        ----------
        df : pandas DataFrame

        Returns
        -------

        '''
        # let's create a class dataframe object first
        self.df = df

        num_uid = len(df.uid.unique())
        num_iid = len(df.iid.unique())

        # create sparse matrices
        self.utility = create_sparse_matrix(df, num_uid, num_iid)
        # only create matrices for test and val if passed
        if test:
            self.test = create_sparse_matrix(test, num_uid, num_iid)
        else:
            self.test = None
        if validation is not None:
            self.validation = create_sparse_matrix(
                validation, num_uid, num_iid)
        else:
            self.validation = None

    def trainModel(self, K=25, epochs=99, gamma=20, lr=0.075):
        ''' 

        Parameters
        ----------

        Returns
        -------

        TODO
        ----


        NOTE: optimal K=25, epochs=99, gamma=20, lr=0.075. Still optimizing but this is ok for initial deployment
        NOTE: I've removed momentum for now. Sloppy but I don't need it right now.
        '''
        # this initializes some embedding matrices
        num_uid = self.utility.shape[0]
        num_iid = self.utility.shape[1]
        self.user_features = create_embeddings(num_uid, K=K, gamma=gamma)
        self.item_features = create_embeddings(num_iid, K=K, gamma=gamma)

        # now perform GD, check if we passed a validation set as well.
        if self.validation is not None:
            self.emb_user, self.emb_item, cost_train, cost_val = gradient_descent(df=self.df,
                                                                                  utility=self.utility,
                                                                                  user_features=self.user_features,
                                                                                  item_features=self.item_features,
                                                                                  epochs=epochs,
                                                                                  val=self.validation,
                                                                                  updates=False)
            return (cost_train, cost_val)

        else:
            self.emb_user, self.emb_item, cost_train = gradient_descent(df=self.df,
                                                                        utility=self.utility,
                                                                        user_features=self.user_features,
                                                                        item_features=self.item_features,
                                                                        epochs=epochs,
                                                                        updates=False)
            return (cost_train,)

    
    def getPredictions(self, df, num_predict=10):
        ''' 
        This is actually more complex than this. By just passing self.df, you're only getting predictions on items already read. 
        What you need to do is get this to predict on each user. To achieve this, create a dataframe with all books for each new user!
        Parameters
        ----------

        Returns
        -------

        TODO
        ----

        '''
        totalPredictions = predict(df=df,
                                   user_features=self.user_features,
                                   item_features=self.item_features)
        grouped = totalPredictions.groupby('uid')
        truncatedPredictions = pd.DataFrame()
        for group in grouped:
            temp = group[1].sort_values(by='prediction',
                                        ascending=False)
            truncatedPredictions = truncatedPredictions.append(
                temp.iloc[:num_predict])

        return truncatedPredictions

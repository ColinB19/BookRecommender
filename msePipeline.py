"""
author: Colin Bradley
last updated: 02/17/2021

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

    def split_test_train(self,
                         testTrainFrac=0.5,
                         ratingsWithheldFrac=0.4,
                         testValFrac=0.5,
                         ratingsThresh=0):
        '''
        This function takes in the entire ratings dataset and splits the interaction matrices into a training, testing and validation set.
        First the function splits off a portion of reviews that are then split up into test and validation sets. This function is different
        from scikit-learn in that it preserves all user-ids in the training set to avoid a cold start problem when testing your algorithm. 

        Parameters
        ----------
        testTrainFrac : float, optional
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

        TODO
        ----
        1. Create cross validation functionality where we can split the data into 3-5 sets.
        '''
        import random
        random.seed(1001)

        # only look at ratings above ratingsThresh
        self.archived_ratings = self.archived_ratings.query(
            'rating >= @ratingsThresh')

        # let's get the list of users for the test set
        uids = self.archived_ratings.uid.unique().tolist()
        test_uids = random.sample(uids, k=int(len(uids)*testTrainFrac))

        # get all data for test users and put the rest in training
        test_users = self.archived_ratings.query('uid == @test_uids')
        train = self.archived_ratings.query('uid != @test_uids')

        # only consider 40% of test users ratings and put the rest in the train set.
        # This prevents the cold start problem on the test set. We will later incorporate
        # new users.
        #
        # This just samples ratingsWithheldFrac of the books
        test = test_users.groupby('uid').sample(
            frac=ratingsWithheldFrac, random_state=1)
        # Now let's sample some of these for a validation set, this could be better but I want to get this working for now
        validation = test.groupby('uid').sample(
            frac=testValFrac, random_state=888)
        # lets add the non train/val books to the test set.
        train = train.append(test_users.drop(test.index), ignore_index=True)
        # now just dropping val set from train
        test.drop(validation.index, inplace=True)

        trainNumBooks = len(train.iid.unique())
        testNumBooks = len(test.iid.unique())
        valNumBooks = len(validation.iid.unique())
        trainNumUsers = len(train.uid.unique())
        testNumUsers = len(test.uid.unique())
        valNumUsers = len(validation.uid.unique())

        print(
            f"The number of books in the train set: {trainNumBooks}, test set: {testNumBooks}, val set: {valNumBooks}. The number of users in the train set: {trainNumUsers}, test set: {testNumUsers}, val set: {valNumUsers}.")

        return train, test, validation


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


def sample_hyperparameters():
    ''' 
    This function returns a random value for each hyperparameter for MSE gradient descent. 
    '''
    return {
        "K": np.random.randint(10, 20),
        "lr": np.random.normal(0.05, 0.025),
        "beta": np.random.normal(0.9, 0.05),
        "gamma": np.random.randint(5, 15),
        "epochs": np.random.randint(50, 80)
    }


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

    def trainModel(self, K=25, epochs=155, gamma=20, lr=0.05):
        ''' 

        Parameters
        ----------

        Returns
        -------

        TODO
        ----


        NOTE: optimal K=25, epochs=155, gamma=20, lr=0.05. Still optimizing but this is ok for initial deployment
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

    def paramSearch(self, num_samples=5):
        ''' 

        Parameters
        ----------

        Returns
        -------

        TODO
        ----

        '''
        hyperparams = pd.DataFrame()
        print('Searching for optimal parameters...')
        for i in range(num_samples):
            # get a random sample of hyperparameters
            params = sample_hyperparameters()
            cost = self.trainModel(K=params["K"],
                                   beta=params["beta"],
                                   epochs=params["epochs"],
                                   gamma=params["gamma"],
                                   lr=params["lr"])

            params['train_mse'] = cost[0]
            # FIXME: This also won't work because of
            # val step in trainModel(), len(cost) will
            # never be 2
            if len(cost) == 2:
                params['val_mse'] = cost[1]
            hyperparams = hyperparams.append(params, ignore_index=True)

        return hyperparams.sort_values(by='train_mse')
    
    def gridSearch(self, dfParams):
        Ks = [20,25,30]
        epochs=[100,125,150]
        gammas = [15,20,25]
        lrs = [0.025, 0.05,0.075]
        for k in Ks:
            for gamma in gammas:
                for epoch in epochs:
                    for lr in lrs:
                        dfError = pd.DataFrame()
                        # this initializes some embedding matrices
                        num_uid = self.utility.shape[0]
                        num_iid = self.utility.shape[1]
                        self.user_features = create_embeddings(num_uid, K=k, gamma=gamma)
                        self.item_features = create_embeddings(num_iid, K=k, gamma=gamma)

                        self.emb_user, self.emb_item, cost_train, dfError = gradient_descent(df = self.df,
                                                                                    utility = self.utility,
                                                                                    user_features = self.user_features,
                                                                                    item_features = self.item_features,
                                                                                    epochs=epoch,
                                                                                    learning_rate = lr,
                                                                                    updates=False,
                                                                                    dfError=dfError)
                    dfParams = dfParams.append([[epoch, lr, cost_train]])
                dfError.to_csv(f"AnalyzedData/error_E{epoch}_L{lr}_K{k}_G{gamma}-{pd.to_datetime('today').strftime('%m-%d-%Y')}.csv")
        return dfParams  

    # def gridSearch(self, dfParams):
    #     K = 25
    #     epochs=[200, 225, 250]
    #     gamma = 20
    #     lrs = [0.01, 0.05]
        
    #     for epoch in tqdm(epochs):
    #         for lr in lrs:
    #             dfError = pd.DataFrame()
    #             # this initializes some embedding matrices
    #             num_uid = self.utility.shape[0]
    #             num_iid = self.utility.shape[1]
    #             self.user_features = create_embeddings(num_uid, K=K, gamma=gamma)
    #             self.item_features = create_embeddings(num_iid, K=K, gamma=gamma)
                
    #             self.emb_user, self.emb_item, cost_train, dfError = gradient_descent(df = self.df,
    #                                                                         utility = self.utility,
    #                                                                         user_features = self.user_features,
    #                                                                         item_features = self.item_features,
    #                                                                         epochs=epoch,
    #                                                                         learning_rate = lr,
    #                                                                         updates=False,
    #                                                                         dfError=dfError)
    #             dfParams = dfParams.append([[epoch, lr, cost_train]])
    #             dfError.to_csv(f"AnalyzedData/error_E{epoch}_L{lr}.csv")
    #     return dfParams  

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

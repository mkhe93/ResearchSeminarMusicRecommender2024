from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype
import pandas as pd
import numpy as np
import csv

def load_data(filename: str, rows: int=None, dataset="") -> pd.DataFrame:
    """
    Reads the filename as a pandas dataframe.
    
    Attributes:
    ----------
    filename: str
        the filename to read
    rows: int
        the number of entries to read from the file (default is `None` and reads the entire file)

    Returns:
    ----------
    df: pd.DataFRame
        pandas dataframe with the user-item interactions
    """
    with open(filename, 'r', encoding="utf-16") as f:
        objects = csv.reader(f, delimiter="\t")
        
        if rows == None:
            columns_list = list(objects)
        else:
            columns_list = [next(objects) for i in range(rows + 1)]

        if dataset == "MSD":
            df = pd.DataFrame(columns_list, columns=['userID', 'itemID'])
            if  any(df.columns != ['userID', 'itemID']):
                df.columns = ['userID', 'itemID']
            df['userID'] = df['userID']
            df['itemID'] = df['itemID']
        elif dataset == "real":
            df = pd.DataFrame(columns_list[1:], columns=columns_list[0])
            df = df.iloc[:,:2]
            if  any(df.columns != ['userID', 'itemID']):
                df.columns = ['userID', 'itemID']
            df['userID'] = df['userID'].astype(np.int64)
            df['itemID'] = df['itemID'].astype(np.int64)

    return df

def create_sparse_matrix(df: pd.DataFrame, dataset=""):
    """
    Generates a sparse matrix from ratings dataframe.
    
    Attributes:
    ----------
    df: pd.DataFrame
        pandas dataframe containing 3 columns (userId, movieId, rating)

    Returns:
    ----------
    X: scipy.sparse.csr_matrix
        sparse user-item interaction matrix
    user_index: dict
        dict that maps user id's to user indices
    item_index: dict
        dict that maps item id's to item indices
    """
    M = df['userID'].nunique()
    N = df['itemID'].nunique()
    if dataset == "MSD":
        users = df["userID"].unique()
        items = df["itemID"].unique()

        # Create indices for users and movies
        user_cat = CategoricalDtype(categories=sorted(users))
        tracks_cat = CategoricalDtype(categories=sorted(items))
        user_index = df["userID"].astype(user_cat)
        item_index = df["itemID"].astype(tracks_cat)

    else:
        users = df["userID"].astype("Int64").unique()
        items = df["itemID"].astype("Int64").unique()

        # Create indices for users and movies
        user_cat = CategoricalDtype(categories=sorted(users))
        tracks_cat = CategoricalDtype(categories=sorted(items))
        user_index = df["userID"].astype("Int64").astype(user_cat)
        item_index = df["itemID"].astype("Int64").astype(tracks_cat)

    X = csr_matrix((np.ones(len(user_index)), (user_index.cat.codes, item_index.cat.codes)), shape=(M,N))
    
    return X, user_index, item_index

def train_test_split(user_item_sparse_csr, user_index, item_index, train_percentage=0.8, k=5, split_strategy=None):
    """ Randomly splits the ratings matrix into two matrices for training/testing.

    Parameters
    ----------
    user_item_sparse_csr : csr_matrix
        A sparse matrix to split
    train_percentage : float
        What percentage of ratings should be used for training
    random_state : int, None or RandomState
        The existing RandomState. If None, or an int, will be used
        to seed a new numpy RandomState.
    Returns
    -------
    (train, test) : csr_matrix, csr_matrix
        A tuple of csr_matrices for training/testing """

    if split_strategy is None:
        random_state = np.random.default_rng(split_strategy)
        random_index = random_state.random(len(user_item_sparse_csr.data))
        train_index = random_index < train_percentage
        test_index = random_index >= train_percentage
    elif split_strategy == "last":
        indeces = np.arange(len(user_item_sparse_csr.data))
        cut_index = int(np.round(indeces.shape[0] * train_percentage))
        train_index = indeces < cut_index
        test_index = indeces >= cut_index
    elif split_strategy == "cross-fold":
        folds = np.linspace(0,1,num=k+1)
        indeces = np.arange(len(user_item_sparse_csr.data))
        cut_indices = [(int(np.round(indeces.shape[0] * folds[i])), int(np.round(indeces.shape[0] * folds[i+1]))) for i in range(k)]
        train_indices = [(indeces >= cut_index[0]) & (indeces < cut_index[1]) for cut_index in cut_indices]
        test_indices = [(indeces < cut_index[0]) | (indeces >= cut_index[1]) for cut_index in cut_indices]

        train_list = [csr_matrix((user_item_sparse_csr.data[train_index],
                            (user_index.cat.codes[train_index], item_index.cat.codes[train_index])),
                        shape=user_item_sparse_csr.shape, dtype=user_item_sparse_csr.dtype) for train_index in train_indices]

        test_list = [csr_matrix((user_item_sparse_csr.data[test_index],
                        (user_index.cat.codes[test_index], item_index.cat.codes[test_index])),
                        shape=user_item_sparse_csr.shape, dtype=user_item_sparse_csr.dtype) for test_index in test_indices]
        
        for test in test_list:
            test.data[test.data < 0] = 0
            test.eliminate_zeros()

        return train_list, test_list
    else:
        random_state = np.random.default_rng(split_strategy)
        random_index = random_state.random(len(user_item_sparse_csr.data))
        train_index = random_index < train_percentage
        test_index = random_index >= train_percentage

    train = csr_matrix((user_item_sparse_csr.data[train_index],
                        (user_index.cat.codes[train_index], item_index.cat.codes[train_index])),
                       shape=user_item_sparse_csr.shape, dtype=user_item_sparse_csr.dtype)

    test = csr_matrix((user_item_sparse_csr.data[test_index],
                       (user_index.cat.codes[test_index], item_index.cat.codes[test_index])),
                      shape=user_item_sparse_csr.shape, dtype=user_item_sparse_csr.dtype)

    test.data[test.data < 0] = 0
    test.eliminate_zeros()

    return train, test
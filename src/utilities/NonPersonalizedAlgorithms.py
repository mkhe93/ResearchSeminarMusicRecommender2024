import pandas as pd
import numpy as np

class NonPersonalizedAlgorithms:
    @staticmethod
    def most_popular(user_ids:list, user_index, item_index, top_N, csr_matrix, already_interacted=[]):

        userIDX = [user_index.cat.codes[user_index==user].unique()[0] for user in user_ids]

        scores = csr_matrix.getnnz(axis=0) / csr_matrix.getnnz(axis=0).max()

        ids = np.argsort(-scores, axis=0)
        scores = np.take_along_axis(scores, ids, axis=0)

        ids = [ids for i in range(len(user_ids))]
        scores = [scores for i in range(len(user_ids))]

        # remove already seen items and extract top_n
        if already_interacted != []:
            condition = [np.isin(ids[i],already_interacted[i], assume_unique=True, invert=True) for i in range(len(user_ids))]
            ids = [ids[i][cond][-top_N:] for i, cond in enumerate(condition)]
            scores = [scores[i][cond][-top_N:] for i, cond in enumerate(condition)]
        else:
            already_interacted = [csr_matrix[user].indices for user in userIDX]
            condition = [np.isin(ids[i],already_interacted[i], assume_unique=True, invert=True) for i in range(len(user_ids))]
            ids = [ids[i][cond][:top_N] for i, cond in enumerate(condition)]
            scores = [scores[i][cond][:top_N] for i, cond in enumerate(condition)]

        recommendations = pd.DataFrame([{
            "itemID": item_index.cat.categories[ids].unique().to_list(),
            "score": scores,
            "userID": userids} for ids, scores, userids in zip(ids, scores, user_ids)]
            )
        
        # sorted on rid
        return recommendations

class NonPersonalizedRecommender:
    def __init__(self, algorithm_class):
        self.name = str
        self.algorithm_class = algorithm_class
        self.algorithm = {name: getattr(algorithm_class, name) for name in dir(algorithm_class) if callable(getattr(algorithm_class, name))}

    def add_algorithm(self, name):
        """
        Adds a new evaluation metric method to the class.
        
        :param name: Name of the metric method.
        :param func: The function that calculates the metric.
        """
        self.name = name
        if name in self.algorithm:
            setattr(self, name, self.algorithm[name])
        else:
            raise ValueError(f"Algorithm '{name}' not found in {self.algorithm_class.__name__}.")
        
    def fit(self, user_item_matrix):
        """
        Fit the model using the user-item interaction matrix.
        
        Parameters:
        user_item_matrix (csr_matrix): User-item interaction matrix (binary ratings).
        """
        if self.name == "most_popular":
            self.user_item_matrix = user_item_matrix

    def recommend(self, user_ids:list, user_index, item_index, top_N, already_interacted=[], *args, **kwargs):
        """
        Evaluates a specific metric.
        
        :param metric_name: Name of the metric to evaluate.
        :param args: Positional arguments to pass to the metric function.
        :param kwargs: Keyword arguments to pass to the metric function.
        :return: The result of the metric function.
        """
        if self.name == "most_popular":
            recommendations = self.algorithm[self.name](user_ids, user_index, item_index, top_N, csr_matrix=self.user_item_matrix, already_interacted=already_interacted)

        return recommendations
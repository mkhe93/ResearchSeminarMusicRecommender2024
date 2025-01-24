import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.ann.annoy import AnnoyModel

class MFAlgorithms:
    @staticmethod
    def als_algorithm(user_ids:list, user_index, item_index, model, top_N, csr_matrix):

        userIDX = [user_index.cat.codes[user_index==user].unique()[0] for user in user_ids]

        ids, scores = model.recommend(
            userIDX, 
            csr_matrix[userIDX],
            filter_already_liked_items=True,
            N=top_N
        )

        recommendations = pd.DataFrame([{
            "itemID": item_index.cat.categories[ids].unique().to_list(),
            "score": scores,
            "userID": userids} for ids, scores, userids in zip(ids, scores, user_ids)]
            )
        
        return recommendations
    
    @staticmethod
    def als_annoy_algorithm(user_ids:list, user_index, item_index, model, top_N, csr_matrix):
        """A fast implementation of the above ALS algortihm using indexing by Annoy."""

        userIDX = [user_index.cat.codes[user_index==user].unique()[0] for user in user_ids]

        ids, scores = model.recommend(
            userIDX, 
            csr_matrix[userIDX],
            filter_already_liked_items=True,
            N=top_N
        )

        recommendations = pd.DataFrame([{
            "itemID": item_index.cat.categories[ids].unique().to_list(),
            "score": scores,
            "userID": userids} for ids, scores, userids in zip(ids, scores, user_ids)]
            )
        
        return recommendations

class MatrixFactorizationRecommender:
    def __init__(self, algorithm_class):
        self.name = str
        self.algorithm_class = algorithm_class
        self.algorithm = {name: getattr(algorithm_class, name) for name in dir(algorithm_class) if callable(getattr(algorithm_class, name))}

    def fit(self, user_item_matrix, factors, regularization, random_state=42, alpha=0.5, iterations=15, learning_rate=0.01, trees=50, pretrain=False):
        """
        Fit the model using the user-item interaction matrix.
        
        Parameters:
        user_item_matrix (csr_matrix): User-item interaction matrix (binary ratings).
        """
        if self.name == "als_algorithm":
            self.train_user_item_matrix = user_item_matrix
            self.model = AlternatingLeastSquares(factors=factors, regularization=regularization, alpha=alpha, iterations=iterations, random_state=random_state)
            self.model.fit(self.train_user_item_matrix)
        elif self.name =="als_annoy_algorithm":
            self.train_user_item_matrix = user_item_matrix
            self.model = AlternatingLeastSquares(factors=factors, regularization=regularization, alpha=alpha, iterations=iterations, random_state=random_state)
            if pretrain:
                self.model.fit(self.train_user_item_matrix)
            self.model = AnnoyModel(self.model, n_trees=trees)
            self.model.fit(self.train_user_item_matrix)


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

    def recommend(self, user_ids:list, test_user_item_matrix, user_index, item_index, top_N, *args, **kwargs):
        """
        Evaluates a specific metric.
        
        :param metric_name: Name of the metric to evaluate.
        :param args: Positional arguments to pass to the metric function.
        :param kwargs: Keyword arguments to pass to the metric function.
        :return: The result of the metric function.
        """
        if self.name == "als_algorithm":
            recommendations = self.algorithm[self.name](user_ids, user_index, item_index, model=self.model, top_N=top_N, csr_matrix=test_user_item_matrix)
        elif self.name == "als_annoy_algorithm":
            recommendations = self.algorithm[self.name](user_ids, user_index, item_index, model=self.model, top_N=top_N, csr_matrix=test_user_item_matrix)

        return recommendations
import pandas as pd
import numpy as np

class Metrics:
    @staticmethod
    def MatchCount(recommended_items, test_set, user_index, item_index, userId):
        userIDX = user_index.cat.codes[user_index==userId].unique()[0]
        user_vector = test_set[userIDX]
        test_items = item_index.cat.categories[user_vector.indices].unique()
        matches = len(np.intersect1d(recommended_items, test_items))

        return matches
    
    @staticmethod
    def Precision(recommended_items, test_set, user_index, item_index, userId):
        userIDX = user_index.cat.codes[user_index==userId].unique()[0]
        user_vector = test_set[userIDX]
        test_items = item_index.cat.categories[user_vector.indices].unique()
        matches = len(np.intersect1d(recommended_items, test_items))

        return matches / len(recommended_items)

    @staticmethod
    def MR(recommended_items, test_set, user_index, item_index, userId):
        userIDX = user_index.cat.codes[user_index==userId].unique()[0]
        user_vector = test_set[userIDX]
        test_items = item_index.cat.categories[user_vector.indices].unique()
        matches = len(np.intersect1d(recommended_items, test_items))

        if matches > 0:
            matches = 1

        return matches

    @staticmethod
    def MAP(recommended_items, test_set, user_index, item_index, userId):
        userIDX = user_index.cat.codes[user_index==userId].unique()[0]
        user_vector = test_set[userIDX]
        test_items = item_index.cat.categories[user_vector.indices].unique()
        matches = np.isin(recommended_items, test_items)
        matches_idx = np.where(matches == 1)[0]

        if len(matches_idx) > 0:
            map = np.mean([1 / (idx + 1) for idx in matches_idx])
        else:
            map = 0

        return map

    @staticmethod
    def MRR(recommended_items, test_set, user_index, item_index, userId):
        userIDX = user_index.cat.codes[user_index==userId].unique()[0]
        user_vector = test_set[userIDX]
        test_items = item_index.cat.categories[user_vector.indices].unique()
        matches = np.isin(recommended_items, test_items)
        first_match = np.where(matches == 1)[0]

        if len(first_match) > 0:
            mrr = np.mean(1 / (first_match[0] + 1))
        else:
            mrr = 0
        return mrr
    

    @staticmethod
    def NDCG(recommended_items, test_set, user_index, item_index, userId):
        """
        Compute the Normalized Discounted Cumulative Gain (NDCG) at rank k.
        
        Args:
            recommended_items (list): List of recommended items.
            relevant_items (set): Set of relevant items.
            k (int): Rank position up to which the NDCG is calculated.
        
        Returns:
            float: NDCG value at rank k.
        """
        def DCG(relevant_items_real_index):
            """
            Compute the Discounted Cumulative Gain (DCG) at rank k.
            
            Args:
                recommended_items (list): List of recommended items.
                relevant_items (set): Set of relevant items.
                k (int): Rank position up to which the DCG is calculated.
            
            Returns:
                float: DCG value at rank k.
            """
            dcg = 0.0
            dcg = np.sum([1 / np.log2(idx + 2) for idx in relevant_items_real_index])
            return dcg
    
        def IDCG(relevant_items_ideal_index):
            """
            Compute the Ideal Discounted Cumulative Gain (IDCG) at rank k.
            
            Args:
                relevant_items (set): Set of relevant items.
                k (int): Rank position up to which the IDCG is calculated.
            
            Returns:
                float: IDCG value at rank k.
            """
            idcg = 0.0
            idcg = np.sum([1 / np.log2(idx + 2) for idx in range(len(relevant_items_ideal_index))])

            return idcg
        
        userIDX = user_index.cat.codes[user_index==userId].unique()[0]
        user_vector = test_set[userIDX]
        relevant_items = item_index.cat.categories[user_vector.indices]

        matches = np.isin(recommended_items, relevant_items)
        matches_idx = np.where(matches == 1)[0]

        if len(matches_idx) > 0:
            dcg = DCG(matches_idx)
            idcg = IDCG(matches_idx)
            ndcg = dcg / idcg if idcg > 0 else 0.0
        else:
            ndcg = 0

        return ndcg
    
    @staticmethod
    def Coverage(recommendations_dict, interaction_matrix):
        """
        Compute the Normalized Discounted Cumulative Gain (NDCG) at rank k.
        
        Args:
            recommended_items (list): List of recommended items.
            relevant_items (set): Set of relevant items.
            k (int): Rank position up to which the NDCG is calculated.
        
        Returns:
            float: NDCG value at rank k.
        """
        total_number_items = interaction_matrix.shape[1]
        
        unique_recommended_items = set()
        for user, items in recommendations_dict.itemID.items():
            unique_recommended_items.update(items)

        number_unique_recommended_items = len(unique_recommended_items)

        return number_unique_recommended_items / total_number_items
    
    @staticmethod
    def APLT(recommended_items, interaction_matrix, item_index, threshold="Median"):
        """
        Calculate the Average Percentage of Long Tail Items (APLT).

        Args:
            recommendations_dict (dict): Dictionary where keys are user IDs and values are lists of recommended item IDs.
            interaction_matrix (csr_matrix): User-item interaction matrix.
            threshold (float): Proportion threshold to determine long-tail items (default is median).

        Returns:
            float: Average Percentage of Long Tail Items (APLT).
        """
        # Determine the median interaction count
        item_popularity = interaction_matrix._getnnz(axis=0)

        if threshold == "Median":
            threshold = np.median(item_popularity)
        else:
            index = np.argsort(-item_popularity)
            index = index[int(len(index) * threshold)]
            threshold = item_popularity[index]

        itemIDX = [item_index.cat.codes[item_index == item].unique() for item in recommended_items]
        recommended_item_popularity = interaction_matrix._getnnz(axis=0)[itemIDX]

        aplt = np.sum(recommended_item_popularity <= threshold) / len(recommended_items)
        
        return aplt

    @staticmethod
    def ARP(recommended_items, interaction_matrix, item_index):
        """
        Calculate the Average Recommendation Popularity (ARP).

        Args:
            recommendations_dict (dict): Dictionary where keys are user IDs and values are lists of recommended item IDs.
            interaction_matrix (csr_matrix): User-item interaction matrix.
            threshold (float): Proportion threshold to determine long-tail items (default is 20%).

        Returns:
            float: Average Recommendation Popularity (ARP).
        """
        itemIDX = [item_index.cat.codes[item_index == item].unique() for item in recommended_items]
        item_popularity = interaction_matrix._getnnz(axis=0)[itemIDX]
        
        arp = np.sum(item_popularity) / len(recommended_items)

        return arp

class Evaluation:
    def __init__(self, metrics_class, csr_matrix):
        self.metrics_class = metrics_class
        self.metrics = {name: getattr(metrics_class, name) for name in dir(metrics_class) if callable(getattr(metrics_class, name))}
        self.user_item_csr = csr_matrix

    def add_metric(self, name):
        """
        Adds a new evaluation metric method to the class.
        
        :param name: Name of the metric method.
        :param func: The function that calculates the metric.
        """
        if name in self.metrics:
            setattr(self, name, self.metrics[name])
        else:
            raise ValueError(f"Metric '{name}' not found in {self.metrics_class.__name__}.")

    def evaluate(self, metric_name, recommendations, test_set, user_index, item_index, threshold="Median", *args, **kwargs):
        """
        Evaluates a specific metric.
        
        :param metric_name: Name of the metric to evaluate.
        :param args: Positional arguments to pass to the metric function.
        :param kwargs: Keyword arguments to pass to the metric function.
        :return: The result of the metric function.
        """
        if metric_name in ["Precision", "MR", "MRR", "MAP", "NDCG"]:
            K = len(recommendations.itemID[0])
            metric = recommendations.apply(lambda x: self.metrics[metric_name](recommended_items=x.itemID, test_set=test_set, user_index=user_index, item_index=item_index, userId=x.userID), axis=1)
            return pd.DataFrame({f"{metric_name}@{K}": metric}).mean().values[0]
        elif metric_name in ["MatchCount"]:
            K = len(recommendations.itemID[0])
            metric = recommendations.apply(lambda x: self.metrics[metric_name](recommended_items=x.itemID, test_set=test_set, user_index=user_index, item_index=item_index, userId=x.userID), axis=1)
            return pd.DataFrame({f"{metric_name}@{K}": metric}).sum().values[0]
        elif metric_name in ["Coverage"]:
            K = len(recommendations.itemID[0]) * recommendations
            metric = self.metrics[metric_name](recommendations_dict=recommendations, interaction_matrix=self.user_item_csr)
            return pd.DataFrame({f"{metric_name}@{K}": metric}, index=[0]).values[0][0]
        elif metric_name in ["APLT"]:
            K = len(recommendations.itemID[0])
            metric = recommendations.apply(lambda x: self.metrics[metric_name](recommended_items=x.itemID, interaction_matrix=self.user_item_csr, item_index=item_index, threshold=threshold), axis=1)
            return pd.DataFrame({f"{metric_name}@{K}": metric}).mean().values[0]
        elif metric_name in ["ARP"]:
            K = len(recommendations.itemID[0])
            metric = recommendations.apply(lambda x: self.metrics[metric_name](recommended_items=x.itemID, interaction_matrix=self.user_item_csr, item_index=item_index), axis=1)
            return pd.DataFrame({f"{metric_name}@{K}": metric}).mean().values[0]
        else:
            raise ValueError(f"Metric '{metric_name}' not found in {self.metrics_class.__name__}.")       
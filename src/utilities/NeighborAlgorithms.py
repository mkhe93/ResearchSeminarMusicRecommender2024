import pandas as pd
import numpy as np

class NeighborhoodAlgorithms:
    @staticmethod
    def user_based_neighborhood(user_ids:list, user_index, item_index, top_N, neighborhood_size, csr_matrix, alpha, q, already_interacted=[]):
        
        if neighborhood_size is None  or neighborhood_size==0:
            neighborhood_size = csr_matrix.shape[0]
                
        # DEPRECATED!
        def calculate_similarity(user_id, csr_matrix, alpha, q):
            """Calculate the asymmetric cosine similarity matrix using matrix operations."""
            dot_product = csr_matrix[user_id].dot(csr_matrix.T).toarray()
            
            # Compute the norms of each user vector
            norms = csr_matrix.getnnz(axis=1)
            
            # Calculate asymmetric cosine similarity
            with np.errstate(divide='ignore', invalid='ignore'):
                p_ab = norms[:, np.newaxis] ** alpha
                p_ba = norms[np.newaxis, :] ** (1 - alpha)
            
            similarity_matrix = dot_product / (p_ab[user_id] * p_ba)
            np.fill_diagonal(similarity_matrix, 0)  # Set the diagonal to 0 to ignore self-similarity
            
            return (similarity_matrix**q)[0]
        
        def calculate_similarity(user_id, csr_matrix, alpha, q):

            dot_product = csr_matrix[user_id].dot(csr_matrix.T).toarray()
            
            # Compute the norms of each user vector
            norms = csr_matrix.getnnz(axis=1)
            
            # Calculate asymmetric cosine similarity
            p_ab = norms[:, np.newaxis] ** alpha
            p_ba = norms[np.newaxis, :] ** (1 - alpha)

            with np.errstate(divide='ignore', invalid='ignore'):
                similarity_matrix = dot_product / (p_ab[user_id] * p_ba)
                similarity_matrix[np.isnan(similarity_matrix)] = 0

            return (similarity_matrix**q)

        # from userID to userIndex in the sparse matrix
        userIDX = [user_index.cat.codes[user_index==user].unique()[0] for user in user_ids]

        # get the row of the user with userID
        user_vector = calculate_similarity(userIDX, csr_matrix, alpha, q)

        # most similar top_N users
        top_M_indices = [vector.argsort()[-neighborhood_size:] for vector in user_vector]
        top_users_similarity = [np.sort(vector)[-neighborhood_size:] for vector in user_vector]

        # aggregated scores to all items by top_M users
        scores = [(user_similarity @ csr_matrix[indices]) / user_similarity.sum() for user_similarity, indices in zip(top_users_similarity, top_M_indices)]
        ids = [np.argsort(-score) for score in scores]
        scores = [np.sort(score)[::-1] for score in scores]  

        # remove already seen items and extract top_n
        if already_interacted != []:
            condition = [np.isin(ids[user_nr],already_interacted[user_nr], assume_unique=True, invert=True) for user_nr in range(len(user_ids))]
        else:
            already_interacted = [csr_matrix[user].indices for user in userIDX]
            condition = [np.isin(ids[user_nr],already_interacted[user_nr], assume_unique=True, invert=True) for user_nr in range(len(user_ids))]
            
        ids = [ids[i][cond][:top_N] for i, cond in enumerate(condition)]
        scores = [scores[i][cond][:top_N] for i, cond in enumerate(condition)]
            
        recommendations = pd.DataFrame([{
            "itemID": item_index.cat.categories[ids].unique().to_list(),
            "score": scores,
            "userID": userids} for ids, scores, userids in zip(ids, scores, user_ids)]
            )

        # sorted on rid
        return recommendations
    
    @staticmethod
    def user_based_iterative_asym_neighborhood(user_ids:list, user_index, item_index, top_N, neighborhood_size, csr_matrix, beta, alpha, q):
        """
        Predict the score matrix for a list of users and all items using the top_n_neighbors most similar users.
        
        Parameters:
        user_ids (list): List of user indices.
        top_n_neighbors (int): Number of top similar users to consider.
        
        Returns:
        np.ndarray: Score matrix of shape (len(user_ids), n_items).
        """
        if neighborhood_size is None or neighborhood_size==0:
            neighborhood_size = csr_matrix.shape[0]

        nItems = csr_matrix.shape[1]

        def calculate_similarity(user_id, csr_matrix, alpha, q):
            """Calculate the asymmetric cosine similarity matrix using matrix operations."""
            dot_product = csr_matrix[user_id].dot(csr_matrix.T).toarray()
            
            # Compute the norms of each user vector
            norms = csr_matrix.getnnz(axis=1)
            
            # Calculate asymmetric cosine similarity
            with np.errstate(divide='ignore', invalid='ignore'):
                p_ab = norms[:, np.newaxis] ** alpha
                p_ba = norms[np.newaxis, :] ** (1 - alpha)
            
            similarity_matrix = dot_product / (p_ab[user_id] * p_ba)
            np.fill_diagonal(similarity_matrix, 0)  # Set the diagonal to 0 to ignore self-similarity
            
            return (similarity_matrix**q)[0]

        #user_item_matrix = self.user_item_matrix.toarray()

        score_matrix = np.zeros((len(user_ids), nItems))

        for i, user_id in enumerate(user_ids):
            user_idx = user_index.cat.codes[user_index == user_id].unique()[0]

            user_sim = calculate_similarity(user_idx, csr_matrix, alpha, q)

            # Step 1: Calculate the top_n_neighbors most similar users for the current user
            top_similar_users = np.argsort(-user_sim)[:neighborhood_size]

            # Step 2: Get the corresponding similarities
            top_similarities = user_sim[top_similar_users]
            top_user_item_norms = np.linalg.norm(top_similarities) ** (2 * beta)

            # Step 3: Mask for the user's rated items
            rated_mask = csr_matrix[user_idx].indices

            # Step 4: Compute numerator and denominator using matrix operations

            with np.errstate(divide='ignore', invalid='ignore'):
                numerator = top_similarities @ csr_matrix[top_similar_users]
                denominator = (top_user_item_norms * csr_matrix.getnnz(axis=0) ** (1 - beta)) / len(user_ids)
                denominator[np.isnan(denominator)] = 0
                user_score = numerator / denominator
                user_score[np.isnan(user_score)] = 0

            # Zero out scores for already seen items
            user_score[rated_mask] = 0

            score_matrix[i] = user_score
        
        # Generate the results using list comprehension
        recommendations = pd.DataFrame([
            {
                "userID": user_id,
                "itemID": item_index.cat.categories[np.argsort(score_matrix[idx])].unique()[-top_N:][::-1].tolist(),
                "score": score_matrix[idx, np.argsort(score_matrix[idx])[-top_N:][::-1]].tolist()
            }
            for idx, user_id in enumerate(user_ids)
        ])
        
        return recommendations

    @staticmethod
    def item_based_iterative_asym_neighborhood(user_ids:list, user_index, item_index, top_N, neighborhood_size, csr_matrix, beta, alpha, q):
        """
        Predict the score matrix for a list of users and all items using the top_n_neighbors most similar users.
        
        Parameters:
        user_ids (list): List of user indices.
        top_n_neighbors (int): Number of top similar users to consider.
        
        Returns:
        np.ndarray: Score matrix of shape (len(user_ids), n_items).
        """
        """
        Predict the score matrix for a list of users and all items using the top_n_neighbors most similar users.
        
        Parameters:
        user_ids (list): List of user indices.
        top_n_neighbors (int): Number of top similar users to consider.
        
        Returns:
        np.ndarray: Score matrix of shape (len(user_ids), n_items).
        """

        if neighborhood_size is None or neighborhood_size==0:
            neighborhood_size = csr_matrix.shape[1]

        def calculate_similarity(item_id, csr_matrix, alpha, q):

            dot_product = csr_matrix.T[item_id].dot(csr_matrix).toarray()
            
            # Compute the norms of each user vector
            norms = csr_matrix.T.getnnz(axis=1)
            
            # Calculate asymmetric cosine similarity
            p_ab = norms[:, np.newaxis] ** alpha
            p_ba = norms[np.newaxis, :] ** (1 - alpha)

            with np.errstate(divide='ignore', invalid='ignore'):
                similarity_matrix = dot_product / (p_ab[item_id] * p_ba)
                similarity_matrix[np.isnan(similarity_matrix)] = 0

            return (similarity_matrix**q)
        
        userIDX = [user_index.cat.codes[user_index==user].unique()[0] for user in user_ids]

        nItems = csr_matrix.shape[1]
        score_matrix = np.zeros((len(user_ids), nItems))

        for i, user_id in enumerate(userIDX):
            rated_items = csr_matrix[user_id, :].indices
            item_sim = calculate_similarity(rated_items, csr_matrix, alpha, q)
        
            # Step 1: Calculate the top neighborhood_size most similar items for each item
            top_similar_items = np.argsort(-item_sim.T)[:,:neighborhood_size]
            # Step 2: Get the corresponding similarities
            top_similarities = np.take_along_axis(item_sim.T, top_similar_items, axis=1)

            item_norms = np.linalg.norm(top_similarities.T, axis=0)

            # Calculate asymmetric cosine similarity
            with np.errstate(divide='ignore', invalid='ignore'):
                item_norms = np.power(item_norms, 2 * (1 - beta))
                item_norms[np.isnan(item_norms)] = 0

            numerator = np.sum(top_similarities, axis=1)
            rated_sums = np.power(csr_matrix[user_id, :].getnnz(axis=1)[0], beta)
            denominator = (item_norms * rated_sums)

            # Step 5: Compute scores
            with np.errstate(divide='ignore', invalid='ignore'):
                user_score = numerator / denominator
                user_score[np.isnan(user_score)] = 0  # Replace NaN values with 0

            # Zero out scores for already seen items
            rated_mask = csr_matrix[user_id].indices
            user_score[rated_mask] = 0

            score_matrix[i] = user_score
                
        # Generate the results using list comprehension
        recommendations = pd.DataFrame([
            {
                "userID": user_id,
                "itemID": item_index.cat.categories[np.argsort(score_matrix[idx])].unique()[-top_N:][::-1].tolist(),
                "score": score_matrix[idx, np.argsort(score_matrix[idx])[-top_N:][::-1]].tolist()
            }
            for idx, user_id in enumerate(user_ids)
        ])
        
        return recommendations 

class NeighborhoodRecommender:
    def __init__(self, algorithm_class):
        self.name = str
        self.algorithm_class = algorithm_class
        self.algorithm = {name: getattr(algorithm_class, name) for name in dir(algorithm_class) if callable(getattr(algorithm_class, name))}

    def fit(self, user_item_matrix, alpha=0.5, q=1):
        """
        Fit the model using the user-item interaction matrix.
        
        Parameters:
        user_item_matrix (csr_matrix): User-item interaction matrix (binary ratings).
        """
        if self.name == "user_based_neighborhood":
            self.train_user_item_matrix = user_item_matrix
            self.n_users, self.n_items = self.train_user_item_matrix.shape            
            self.alpha = alpha
            self.q = q

        elif self.name == "user_based_iterative_asym_neighborhood":
            self.train_user_item_matrix = user_item_matrix
            self.n_users, self.n_items = self.train_user_item_matrix.shape
            self.alpha = alpha
            self.q = q

        elif self.name == "item_based_iterative_asym_neighborhood":
            self.train_user_item_matrix = user_item_matrix
            self.n_users, self.n_items = self.train_user_item_matrix.shape
            self.alpha = alpha
            self.q = q

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

    def recommend(self, user_ids:list, test_user_item_matrix, user_index, item_index, top_N, neighborhood_size, already_interacted=[], beta=1, *args, **kwargs):
        """
        Evaluates a specific metric.
        
        :param metric_name: Name of the metric to evaluate.
        :param args: Positional arguments to pass to the metric function.
        :param kwargs: Keyword arguments to pass to the metric function.
        :return: The result of the metric function.
        """
        if self.name == "user_based_neighborhood":
            recommendations = self.algorithm[self.name](user_ids, user_index, item_index, top_N, neighborhood_size=neighborhood_size, csr_matrix=test_user_item_matrix, alpha=self.alpha, q=self.q, already_interacted=already_interacted)
        elif self.name == "user_based_iterative_asym_neighborhood":
            recommendations = self.algorithm[self.name](user_ids, user_index, item_index, top_N, neighborhood_size=neighborhood_size, csr_matrix=test_user_item_matrix, beta=beta, alpha=self.alpha, q=self.q)
        elif self.name == "item_based_iterative_asym_neighborhood":
            recommendations = self.algorithm[self.name](user_ids, user_index, item_index, top_N, neighborhood_size=neighborhood_size, csr_matrix=test_user_item_matrix, beta=beta, alpha=self.alpha, q=self.q)


        return recommendations
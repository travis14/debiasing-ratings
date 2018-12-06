import numpy as np
import pandas as pd
from scipy.spatial import distance

class RatingsGraph:
	def __init__(self, ratings_graph, ratings, entity_sim=None, ground_truth_ratings=None):
		self.ratings_graph = ratings_graph # rows user, cols entities
		self.ratings = ratings
		if entity_sim:
			self.entity_sim = entity_sim
		if ground_truth_ratings:
			self.ground_truth_ratings = ground_truth_ratings

	def get_graph_shape(self):
		return self.ratings_graph.shape

	def get_graph(self):
		return self.ratings_graph

	def get_ratings(self):
		return self.ratings

	def get_ground_truth_ratings(self):
		return self.ground_truth_ratings

	def get_entity_rating_counts(self):
		return np.sum(self.ratings_graph, axis=0)

	def get_user_rating_counts(self):
		return np.sum(self.ratings_graph, axis=1)
import numpy as np
import pandas as pd

EPSILON = 0.000001

def single_iteration(ratings_graph, biases, true_ratings, alpha):
	# Update Ratings
	graph_shape = ratings_graph.get_graph_shape()
	indiv_true_ratings = np.maximum(np.zeros(graph_shape), np.minimum(np.ones(graph_shape),
										ratings_graph.get_ratings() - alpha * biases[:, None]))
	rating_denoms = ratings_graph.get_entity_rating_counts()
	next_true_ratings = np.sum(ratings_graph.get_graph() * indiv_true_ratings, axis=0) / rating_denoms

	# Update Biases
	indiv_biases = ratings_graph.get_ratings() - next_true_ratings
	bias_denoms = ratings_graph.get_user_rating_counts()
	next_biases = np.sum(ratings_graph.get_graph() * indiv_biases, axis=1) / bias_denoms

	converged = True
	if (true_ratings is not None and np.any(np.abs(true_ratings - next_true_ratings) > EPSILON)) or \
		np.any(np.abs(biases - next_biases) > EPSILON):
		converged = False

	return converged, next_true_ratings, next_biases
	

def debias_ratings_baseline(ratings_graph, alpha, max_iters):
	num_users, num_entities = ratings_graph.get_graph_shape()
	true_ratings = []
	biases = [np.random.rand(num_users)]
	errors = []

	converged = False
	num_iters = 0
	while not converged and num_iters < max_iters:
		true_rate_or_none = None if not true_ratings else true_ratings[-1]
		iter_out = single_iteration(ratings_graph, biases[-1], true_rate_or_none, alpha)
		converged, next_true_ratings, next_biases = iter_out
		true_ratings.append(next_true_ratings)
		biases.append(next_biases)
		print(converged)
		num_iters += 1

	return biases, true_ratings, errors
import numpy as np
import pandas as pd

EPSILON = 0.000001

def single_iteration(ratings_graph, biases, true_ratings, alpha, beta):
	# Update Ratings
	graph_shape = ratings_graph.get_graph_shape()
	indiv_true_ratings = np.maximum(np.zeros(graph_shape), np.minimum(np.ones(graph_shape),
										ratings_graph.get_ratings() - alpha * biases[:, None]))
	rating_denoms = ratings_graph.get_entity_rating_counts()
	next_true_ratings = np.sum(ratings_graph.get_graph() * indiv_true_ratings, axis=0) / rating_denoms

	# Update Biases
	indiv_biases = ratings_graph.get_ratings() - next_true_ratings
	bias_denoms = ratings_graph.get_user_rating_counts()
	next_biases = (1-beta)*biases + beta*(np.sum(ratings_graph.get_graph() * indiv_biases, axis=1) / bias_denoms)

	converged = True
	if ((true_ratings is not None) and np.any(np.abs(true_ratings - next_true_ratings) > EPSILON)) or \
		np.any(np.abs(biases - next_biases) > EPSILON):
		converged = False

	return converged, next_true_ratings, next_biases

def single_iteration_user_ent(ratings_graph, biases, true_ratings, alpha, beta):
	# Update Rating
	graph_shape = ratings_graph.get_graph_shape()
	indiv_true_ratings = np.maximum(np.zeros(graph_shape), np.minimum(np.ones(graph_shape),
										ratings_graph.get_ratings() - alpha * biases))
	rating_denoms = ratings_graph.get_entity_rating_counts()
	next_true_ratings = np.sum(ratings_graph.get_graph() * indiv_true_ratings, axis=0) / rating_denoms

	# Update Biases
	indiv_biases = (ratings_graph.get_graph()*(ratings_graph.get_ratings() - next_true_ratings)).dot(ratings_graph.get_entity_sim())
	bias_denoms = (ratings_graph.get_graph()).dot(ratings_graph.get_entity_sim())
	next_biases = (1-beta)*biases + beta*(indiv_biases) / bias_denoms

	converged = True
	if ((true_ratings is not None) and np.any(np.abs(true_ratings - next_true_ratings) > EPSILON)) or \
		np.any(np.abs(biases - next_biases) > EPSILON):
		converged = False

	return converged, next_true_ratings, next_biases


def debias_ratings_baseline(ratings_graph, initial_alpha, decay_rate, max_iters, beta, user_entity_specific=False):
	np.random.seed(10)
	ground_truth_ratings = ratings_graph.get_ground_truth_ratings()
	true_ratings = [np.random.uniform((ratings_graph.num_entities,))]
	if not user_entity_specific:
		biases = [np.random.uniform(low = -1, high = 1, size = (ratings_graph.num_users,))]
	else:
		biases = [np.random.uniform(low = -1, high = 1, size = (ratings_graph.num_users, ratings_graph.num_entities))]
	errors = []

	converged = False
	num_iters = 0
	alpha = initial_alpha
	while not converged and num_iters < max_iters:
		true_rate_or_none = None if not true_ratings else true_ratings[-1]
		if not user_entity_specific:
			iter_out = single_iteration(ratings_graph, biases[-1], true_rate_or_none, alpha, beta)
		else:
			iter_out = single_iteration_user_ent(ratings_graph, biases[-1], true_rate_or_none, alpha, beta)

		converged, next_true_ratings, next_biases = iter_out
		true_ratings.append(next_true_ratings)
		biases.append(next_biases)
		if ground_truth_ratings is not None:
			errors.append(np.sqrt(np.mean((next_true_ratings - ground_truth_ratings)**2)))
		num_iters += 1
		alpha = alpha/decay_rate

	return biases, true_ratings, errors

def similarity_weights_approach(ratings_graph):
	similarity = ratings_graph.get_entity_sim()
	existing_edges = ratings_graph.get_graph()
	ratings_existing = ratings_graph.get_ratings()*existing_edges

	stacked_ratings_graph = np.stack([ratings_existing]*ratings_existing.shape[-1], axis=1) #n x n x m
	diff_ratings_graph = np.stack([ratings_existing]*ratings_existing.shape[-1], axis=2)
	weights = np.zeros(ratings_existing.shape)
	for i in range(ratings_graph.num_users): #users
	    for j in range(ratings_graph.num_entities): #entities
	        if existing_edges[i,j] > 0:
	            weights[i,j] = sum([similarity[j,k]*abs(ratings_existing[i,j] - ratings_existing[i,k])
	                                for k in range(ratings_existing.shape[1]) if ratings_graph.get_graph()[i,k] > 0])
	weights = weights/np.expand_dims(np.sum(weights*existing_edges, 1), 1)
	#weights = np.tensordot(np.absolute(diff_ratings_graph - stacked_ratings_graph), similarity)
	#weights = np.einsum('ijk,jk->ik', np.absolute(diff_ratings_graph-stacked_ratings_graph), similarity)
	true_ratings = np.sum(np.multiply(weights, ratings_existing), 0)* 1.0/np.sum(np.multiply(weights, ratings_graph.get_graph()), 0)
	return weights, true_ratings

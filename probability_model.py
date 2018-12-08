import numpy as np

EPSILON = 0.000001

def robust_nanprod(a, axis):
	return np.log(np.nansum(np.exp(a), axis=axis))

def logit(p):
	return np.log(p/(1-p))

def cond_prob_rating(rating, true_rating, covariates, params):
	gamma, mu, sigma_sq = params['gamma'], params['mu'], params['sigma_sq']
	mu_r = mu[true_rating]
	norm_const = (rating*(1-rating)*np.sqrt(2*np.pi*sigma_sq))
	return np.exp(-(logit(rating) - covariates.dot(gamma) - mu_r)**2/(2*sigma_sq))/norm_const

def E_step(na_ratings_matrix, covariates, params):
	'''
	ratings_graph: (n, m) matrix with 1 in position (j, i) if user j rates entity i
	ratings_matrix: (n, m) matrix with user j's rating of entity i in position (i,j)
		or 0 if user j didn't rate entity i (note that a user's rating could be 0)
	entity_attrs: (m, d) matrix where row i is a vector of entity i's attributes
	user_attrs: (n, k) matrix where row j is a vector of user j's attributes
	'''
	p = params['p']

	# r = 1
	all_ones = np.ones(na_ratings_matrix.shape, dtype=int)
	cond_prob_ratings_1 = cond_prob_rating(na_ratings_matrix, all_ones, covariates, params)
	p_all_ratings_given_true_1 = robust_nanprod(cond_prob_ratings_1, axis=1)[:, None]

	# r = 0
	all_zeros = np.zeros(na_ratings_matrix.shape, dtype=int)
	cond_prob_ratings_0 = cond_prob_rating(na_ratings_matrix, all_zeros, covariates, params)
	p_all_ratings_given_true_0 = robust_nanprod(cond_prob_ratings_0, axis=1)[:, None]
	
	unnormalized_probs = np.hstack(((1-p)[:, None], p[:, None]))*\
						np.hstack((p_all_ratings_given_true_0, p_all_ratings_given_true_1))
	return unnormalized_probs / np.sum(unnormalized_probs, axis=1)[:, None]

def gradient_cond_rating_prob(na_ratings_matrix, poster_rating_probs, covariates, params):
	gamma, mu, sigma_sq = params['gamma'], params['mu'], params['sigma_sq']
	#print(covariates.shape, params['gamma'].shape)
	rating_diffs = [logit(na_ratings_matrix) - covariates.dot(gamma) - mu[0], 
						logit(na_ratings_matrix) - covariates.dot(gamma) - mu[1]]
	mu_grad = np.array([np.sum(poster_rating_probs[:, r]*np.nansum(rating_diffs[r], axis=1))/sigma_sq \
						for r in range(2)])
	gamma_grad = np.sum(np.array([np.sum(poster_rating_probs[:, r][:, None]*np.nansum(rating_diffs[r][:, :, None]*covariates, axis=1), axis=0)/sigma_sq \
									for r in range(2)]), axis=0)
	
	return np.hstack((mu_grad, gamma_grad))

def hessian_cond_rating_prob(ratings_mask, poster_rating_probs, covariates, params):
	gamma, sigma_sq = params['gamma'], params['sigma_sq']
	sec_deriv_mu = [-np.sum(poster_rating_probs[:, r]*np.sum(ratings_mask, axis=1))/sigma_sq for r in range(2)]

	# TODO: Construct Hessian using numpy functions
	hessian_gamma = np.zeros((gamma.shape[0], gamma.shape[0]))
	for i in range(ratings_mask.shape[0]):
		for j in range(ratings_mask.shape[1]):
			for r in range(2):
				hessian_gamma += ratings_mask[i, j]*poster_rating_probs[i, r]*covariates[i, j][:, None].dot(covariates[i, j][None, :])
	hessian_gamma /= -sigma_sq

	# TODO: Cross gradients using numpy functions
	cross_grads = np.array([np.zeros(gamma.shape[0]), np.zeros(gamma.shape[0])])
	for r in range(2):
		for i in range(ratings_mask.shape[0]):
			for j in range(ratings_mask.shape[1]):
				cross_grads[r] -= ratings_mask[i, j]*poster_rating_probs[i, r]*covariates[i, j]/sigma_sq

	return np.block([[np.diag(sec_deriv_mu), np.vstack(cross_grads)], [cross_grads.T, hessian_gamma]])

def params_not_converged(params, prev_params):
	return np.any(np.hstack([np.abs(params[param_name] - prev_params[param_name]) > EPSILON for param_name in params]))

def copy_params(params):
	return {param_name: param if type(param) == type(1.0) else param.copy() for param_name, param in params.iteritems()}

def fit_probability_model(ratings_graph, ratings_matrix, entity_attrs, user_attrs):
	m, d = entity_attrs.shape
	n, k = user_attrs.shape

	params = {'p': 0.5*np.ones(m), 'gamma': np.ones((3,)), 'mu': np.array([1, 5]), 'sigma_sq': 0.7}

	na_ratings_graph = ratings_graph.copy().astype(float)
	na_ratings_graph[na_ratings_graph == 0] = np.nan
	na_ratings_matrix = (na_ratings_graph * ratings_matrix).T

	entity_attrs_solid = np.broadcast_to(entity_attrs[:, None, :], (m, n, k))
	user_attrs_solid = np.broadcast_to(user_attrs[None, :, :], (m, n, d))
	covariates = np.concatenate((np.ones((m, n, 1)), entity_attrs_solid, user_attrs_solid), axis=2)

	prev_params = copy_params(params)
	num_iters = 0
	while num_iters == 0 or params_not_converged(params, prev_params):
		poster_rating_probs = E_step(na_ratings_matrix, covariates, params)

		prev_params = copy_params(params)
		newton_raphson_iters = 0
		newton_raphson_prev_params = copy_params(params)
		while newton_raphson_iters == 0 or params_not_converged(params, newton_raphson_prev_params):
			#hessian = hessian_cond_rating_prob(ratings_graph.T, poster_rating_probs, covariates, params)
			gradient = gradient_cond_rating_prob(na_ratings_matrix, poster_rating_probs, covariates, params)
			new_params = np.hstack((params['mu'], params['gamma'])) + 0.00001*gradient#- np.linalg.inv(hessian).dot(gradient)
			
			newton_raphson_prev_params = copy_params(params)
			params['mu'] = new_params[:2]
			#print(params['mu'], prev_params['mu'])
			params['gamma'] = new_params[2:]
			#print(params['gamma'], params['gamma'])
			params['p'] = poster_rating_probs[:, 1]
			newton_raphson_iters += 1
			if newton_raphson_iters % 100 == 0:
				print('NR Iterations', newton_raphson_iters, params['gamma'])

		num_iters += 1
		if num_iters % 10 == 0:
			print('EM Iterations:', num_iters, params['gamma'])

	return params
import numpy as np
from hyperopt import hp
import cPickle

from config import feature_prefix, clean_input_prefix, n_run, n_fold, debug


skl_random_seed = 2016

if debug:
    skl_n_jobs = 1
    skl_min_n_estimators = 5
    skl_max_n_estimators = 10
    skl_n_estimators_step = 5
    iter_step = 5
    hyperopt_param = {}
    hyperopt_param['svr_max_evals'] = 1
    hyperopt_param['lr_max_evals'] = 1
    hyperopt_param['ridge_max_evals'] = 1
else:
    skl_n_jobs = 2
    skl_min_n_estimators = 10
    skl_min_n_estimators = 10
    skl_max_n_estimators = 500
    skl_n_estimators_step = 10
    iter_step = 10
    hyperopt_param = {}
    hyperopt_param['svr_max_evals'] = 200
    hyperopt_param['lr_max_evals'] = 200
    hyperopt_param['ridge_max_evals'] = 200


with open('%s/param_spaces.pkl'%(clean_input_prefix), "rb") as f:
	param_spaces = cPickle.load(f)

#########model dictionary
param_space_reg_skl_svr = {
    'C': hp.loguniform("C", np.log(1), np.log(100)),
    'gamma': hp.loguniform("gamma", np.log(0.001), np.log(0.1)),
    'degree': hp.quniform('degree', 1, 5, 1),
    'epsilon': hp.loguniform("epsilon", np.log(0.001), np.log(0.1)),
    'kernel': hp.choice('kernel', ['rbf', 'poly']),
    "max_evals": hyperopt_param["svr_max_evals"],
}
model_name = 'reg_skl_svr'
param_spaces[model_name] = param_space_reg_skl_svr

param_space_reg_skl_lr = {
    'C': hp.loguniform("C", np.log(0.001), np.log(10)),
    'random_state': skl_random_seed,
    "max_evals": hyperopt_param["lr_max_evals"],
}
model_name = 'reg_skl_lr'
param_spaces[model_name] = param_space_reg_skl_lr

param_space_reg_skl_ridge = {
    'alpha': hp.loguniform("alpha", np.log(0.01), np.log(20)),
    'random_state': skl_random_seed,
    "max_evals": hyperopt_param["ridge_max_evals"],
}
model_name = 'reg_skl_ridge'
param_spaces[model_name] = param_space_reg_skl_ridge
###########################

with open('%s/param_spaces.pkl'%(clean_input_prefix), "wb") as f:
	cPickle.dump(param_spaces, f, -1)

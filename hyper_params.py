import numpy as np
from hyperopt import hp
import cPickle

param_spaces = {}
int_feat = []
#########model dictionary
param_space_clf_xgb_tree = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eta': hp.quniform('eta', 0.01, 1, 0.01),
    'gamma': hp.quniform('gamma', 0, 2, 0.1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'max_depth': hp.quniform('max_depth', 1, 10, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.1),
    'num_round': hp.quniform('num_round', 10, 500, 10),
    'nthread': 4,
    'silent': 1,
    'seed': 1916,
}
model_name = 'clf_xgb_tree'
param_spaces[model_name] = param_space_clf_xgb_tree
int_feat += ["num_round",  "max_depth"]
            
param_space_clf_skl_rf = {
    'n_estimators': hp.quniform("n_estimators", 10, 500, 10),
    'max_features': hp.quniform("max_features", 0.05, 1.0, 0.05),
    'n_jobs': 4,
    'random_state': 1916,
}
model_name = 'clf_skl_rf'
param_spaces[model_name] = param_space_clf_skl_rf
int_feat += ["n_estimators"]

param_space_clf_skl_etr = {
    'n_estimators': hp.quniform("n_estimators", 10, 500, 10),
    'max_features': hp.quniform("max_features", 0.05, 1.0, 0.05),
    'n_jobs': 4,
    'random_state': 1916,
}
model_name = 'clf_skl_etr'
param_spaces[model_name] = param_space_clf_skl_etr

param_space_clf_skl_gbm = {
    'n_estimators': hp.quniform("n_estimators", 10, 500, 10),
    'learning_rate': hp.quniform("learning_rate", 0.01, 0.5, 0.01),
    'max_features': hp.quniform("max_features", 0.05, 1.0, 0.05),
    'max_depth': hp.quniform('max_depth', 1, 15, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
    'random_state': 1916, 
}
model_name = 'clf_skl_gbm'
param_spaces[model_name] = param_space_clf_skl_gbm

param_space_clf_skl_lr = {
    'C': hp.loguniform("C", np.log(0.001), np.log(10)),
    'random_state': 1916,
}
model_name = 'clf_skl_lr'
param_spaces[model_name] = param_space_clf_skl_lr
###########################

###############################################################
# Modules import
###############################################################

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 
import csv
import os
import joblib

from matplotlib.cm import viridis
from scipy import stats
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PolynomialFeatures, OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from skopt import BayesSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

###############################################################
# Data import
###############################################################

X_full_test = pd.read_csv("./data/processed/X_full_test.csv")
X_full_train = pd.read_csv("./data/processed/X_full_train.csv")
X_sub_test = pd.read_csv("./data/processed/X_sub_test.csv")
X_sub_train = pd.read_csv("./data/processed/X_sub_train.csv")
y_full_test = pd.read_csv("./data/processed/y_full_test.csv")
y_full_train = pd.read_csv("./data/processed/y_full_train.csv")
y_sub_test = pd.read_csv("./data/processed/y_sub_test.csv")
y_sub_train = pd.read_csv("./data/processed/y_sub_train.csv")

###############################################################
# Initialize models and parameters
###############################################################

elastic_net_full = ElasticNet()
elastic_net_sub = ElasticNet()

kfolds = KFold(n_splits=5, shuffle=True, random_state=42)
elastic_net_bayes_params = {'alpha': (0.0000001, 0.00001), 'l1_ratio': (0.001, 1)}
elastic_net_random_params = {'alpha': np.logspace(-7, -5, 100), 'l1_ratio': np.linspace(0.0001, 1, 100)}
elastic_net_grid_params = {'alpha': [0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001], 'l1_ratio': [0.005, 0.01, 0.05, 0.1, 0.5, 1]}

###############################################################
# Modelling with BayesSearchCV for Hyperparameter tuning
###############################################################

# Use BayesSearchCV with the defined search space
elastic_net_bayes_full = BayesSearchCV(elastic_net_full, search_spaces=elastic_net_bayes_params, n_iter=20, cv=kfolds, n_jobs=-1)
elastic_net_bayes_sub = BayesSearchCV(elastic_net_sub, search_spaces=elastic_net_bayes_params, n_iter=20, cv=kfolds, n_jobs=-1)

# Fit the models
elastic_net_bayes_full.fit(X_full_train, y_full_train)
elastic_net_bayes_sub.fit(X_sub_train, y_sub_train)

# Make predictions
elastic_net_bayes_y_full_pred = elastic_net_bayes_full.predict(X_full_test)
elastic_net_bayes_y_sub_pred = elastic_net_bayes_sub.predict(X_sub_test)

# Inverse transformation for predicted values
elastic_net_bayes_y_full_pred_inverse = np.expm1(elastic_net_bayes_y_full_pred)
elastic_net_bayes_y_sub_pred_inverse = np.expm1(elastic_net_bayes_y_sub_pred)

# Calculate and print evaluation metrics
mse_elastic_net_bayes_full = mean_squared_error(y_full_test, elastic_net_bayes_y_full_pred_inverse)
mse_elastic_net_bayes_sub = mean_squared_error(y_sub_test, elastic_net_bayes_y_sub_pred_inverse)

rmse_elastic_net_bayes_full = np.sqrt(mse_elastic_net_bayes_full)
rmse_elastic_net_bayes_sub = np.sqrt(mse_elastic_net_bayes_sub)

r2_elastic_net_bayes_full = r2_score(y_full_test, elastic_net_bayes_y_full_pred_inverse)
r2_elastic_net_bayes_sub = r2_score(y_sub_test, elastic_net_bayes_y_sub_pred_inverse)

elastic_net_bayes_full_best_params = elastic_net_bayes_full.best_params_
elastic_net_bayes_sub_best_params = elastic_net_bayes_sub.best_params_

###############################################################
# Modelling with RandomizedSearchCV for Hyperparameter tuning
###############################################################

# Use RandomizedSearchCV with the defined search space
elastic_net_random_full = RandomizedSearchCV(elastic_net_full, param_distributions=elastic_net_random_params, n_iter=20, cv=kfolds, n_jobs=-1)
elastic_net_random_sub = RandomizedSearchCV(elastic_net_sub, param_distributions=elastic_net_random_params, n_iter=20, cv=kfolds, n_jobs=-1)

# Fit the models
elastic_net_random_full.fit(X_full_train, y_full_train)
elastic_net_random_sub.fit(X_sub_train, y_sub_train)

# Make predictions
elastic_net_random_y_full_pred = elastic_net_random_full.predict(X_full_test)
elastic_net_random_y_sub_pred = elastic_net_random_sub.predict(X_sub_test)

# Inverse transformation for predicted values
elastic_net_random_y_full_pred_inverse = np.expm1(elastic_net_random_y_full_pred)
elastic_net_random_y_sub_pred_inverse = np.expm1(elastic_net_random_y_sub_pred)

# Calculate and print evaluation metrics
mse_elastic_net_random_full = mean_squared_error(y_full_test, elastic_net_random_y_full_pred_inverse)
mse_elastic_net_random_sub = mean_squared_error(y_sub_test, elastic_net_random_y_sub_pred_inverse)

rmse_elastic_net_random_full = np.sqrt(mse_elastic_net_random_full)
rmse_elastic_net_random_sub = np.sqrt(mse_elastic_net_random_sub)

r2_elastic_net_random_full = r2_score(y_full_test, elastic_net_random_y_full_pred_inverse)
r2_elastic_net_random_sub = r2_score(y_sub_test, elastic_net_random_y_sub_pred_inverse)

elastic_net_random_full_best_params = elastic_net_random_full.best_params_
elastic_net_random_sub_best_params = elastic_net_random_sub.best_params_

###############################################################
# Modelling with GridSearchCV for Hyperparameter tuning
###############################################################

# Use GridSearchCV instead of BayesSearchCV
elastic_net_grid_full = GridSearchCV(elastic_net_full, elastic_net_grid_params, cv=kfolds)
elastic_net_grid_sub = GridSearchCV(elastic_net_sub, elastic_net_grid_params, cv=kfolds)

# Fit the models
elastic_net_grid_full.fit(X_full_train, y_full_train)
elastic_net_grid_sub.fit(X_sub_train, y_sub_train)

# Make predictions
elastic_net_grid_y_full_pred = elastic_net_grid_full.predict(X_full_test)
elastic_net_grid_y_sub_pred = elastic_net_grid_sub.predict(X_sub_test)

# Inverse transformation for predicted values
elastic_net_grid_y_full_pred_inverse = np.expm1(elastic_net_grid_y_full_pred)
elastic_net_grid_y_sub_pred_inverse = np.expm1(elastic_net_grid_y_sub_pred)

# Calculate and print evaluation metrics
mse_elastic_net_grid_full = mean_squared_error(y_full_test, elastic_net_grid_y_full_pred_inverse)
mse_elastic_net_grid_sub = mean_squared_error(y_sub_test, elastic_net_grid_y_sub_pred_inverse)

rmse_elastic_net_grid_full = np.sqrt(mse_elastic_net_grid_full)
rmse_elastic_net_grid_sub = np.sqrt(mse_elastic_net_grid_sub)

r2_elastic_net_grid_full = r2_score(y_full_test, elastic_net_grid_y_full_pred_inverse)
r2_elastic_net_grid_sub = r2_score(y_sub_test, elastic_net_grid_y_sub_pred_inverse)

elastic_net_grid_full_best_params = elastic_net_grid_full.best_params_
elastic_net_grid_sub_best_params = elastic_net_grid_sub.best_params_

###############################################################
# Save results
###############################################################

# Create a list to store the results
results_elastic_net_list = []

# Append results for the full data using Bayes
results_elastic_net_list.append({
    'data': 'full',
    'best_alpha': elastic_net_bayes_full_best_params['alpha'],
    'best_l1_ratio': elastic_net_bayes_full_best_params['l1_ratio'],
    'mse': mse_elastic_net_bayes_full,
    'rmse': rmse_elastic_net_bayes_full,
    'r_square': r2_elastic_net_bayes_full,
    'tuning_method': 'bayes'
})

# Append results for the sub data using Bayes
results_elastic_net_list.append({
    'data': 'sub',
    'best_alpha': elastic_net_bayes_sub_best_params['alpha'],
    'best_l1_ratio': elastic_net_bayes_sub_best_params['l1_ratio'],
    'mse': mse_elastic_net_bayes_sub,
    'rmse': rmse_elastic_net_bayes_sub,
    'r_square': r2_elastic_net_bayes_sub,
    'tuning_method': 'bayes'
})

# Append results for the full data using Random
results_elastic_net_list.append({
    'data': 'full',
    'best_alpha': elastic_net_random_full_best_params['alpha'],
    'best_l1_ratio': elastic_net_random_full_best_params['l1_ratio'],
    'mse': mse_elastic_net_random_full,
    'rmse': rmse_elastic_net_random_full,
    'r_square': r2_elastic_net_random_full,
    'tuning_method': 'random'
})

# Append results for the sub data using Random
results_elastic_net_list.append({
    'data': 'sub',
    'best_alpha': elastic_net_random_sub_best_params['alpha'],
    'best_l1_ratio': elastic_net_random_sub_best_params['l1_ratio'],
    'mse': mse_elastic_net_random_sub,
    'rmse': rmse_elastic_net_random_sub,
    'r_square': r2_elastic_net_random_sub,
    'tuning_method': 'random'
})

# Append results for the full data using Grid
results_elastic_net_list.append({
    'data': 'full',
    'best_alpha': elastic_net_grid_full_best_params['alpha'],
    'best_l1_ratio': elastic_net_grid_full_best_params['l1_ratio'],
    'mse': mse_elastic_net_grid_full,
    'rmse': rmse_elastic_net_grid_full,
    'r_square': r2_elastic_net_grid_full,
    'tuning_method': 'grid'
})

# Append results for the sub data using Grid
results_elastic_net_list.append({
    'data': 'sub',
    'best_alpha': elastic_net_grid_sub_best_params['alpha'],
    'best_l1_ratio': elastic_net_grid_sub_best_params['l1_ratio'],
    'mse': mse_elastic_net_grid_sub,
    'rmse': rmse_elastic_net_grid_sub,
    'r_square': r2_elastic_net_grid_sub,
    'tuning_method': 'grid'
})

# Convert the list of dictionaries to a DataFrame
results_elastic_net = pd.DataFrame(results_elastic_net_list)

# Reorder rows
new_order = [0, 2, 4, 1, 3, 5]
results_elastic_net = results_elastic_net.loc[new_order].reset_index(drop=True)

# Save
results_elastic_net.to_csv(os.path.join("./models/model_evaluation/", "results_elastic_net.csv"), index=False)

###############################################################
# Save best models
###############################################################

joblib.dump(elastic_net_bayes_full, 'elastic_net_bayes_full_model.pkl')
joblib.dump(elastic_net_bayes_sub, 'elastic_net_bayes_sub_model.pkl')
joblib.dump(elastic_net_random_full, 'elastic_net_random_full_model.pkl')
joblib.dump(elastic_net_random_sub, 'elastic_net_random_sub_model.pkl')
joblib.dump(elastic_net_grid_full, 'elastic_net_grid_full_model.pkl')
joblib.dump(elastic_net_grid_sub, 'elastic_net_grid_sub_model.pkl')

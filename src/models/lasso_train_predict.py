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
from sklearn.metrics import mean_squared_error,  r2_score
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

lasso_full = Lasso()
lasso_sub = Lasso()

kfolds = KFold(n_splits=5, shuffle=True, random_state=42)
lasso_bayes_params = {'alpha': (1e-6, 1e-4, 'log-uniform')}
lasso_random_params = {'alpha': np.logspace(-6, -4, 100)}
lasso_grid_params = {'alpha': [1e-6, 1e-5, 1e-4]}

###############################################################
# Modelling with BayesSearchCV for Hyperparametertuning
###############################################################

# Use BayesSearchCV with the defined search space
lasso_bayes_full = BayesSearchCV(lasso_full, search_spaces=lasso_bayes_params, n_iter=20, cv=kfolds, n_jobs=-1)
lasso_bayes_sub = BayesSearchCV(lasso_sub, search_spaces=lasso_bayes_params, n_iter=20, cv=kfolds, n_jobs=-1)

# Fit the models
lasso_bayes_full.fit(X_full_train, y_full_train)
lasso_bayes_sub.fit(X_sub_train, y_sub_train)

# Make predictions
lasso_bayes_y_full_pred = lasso_bayes_full.predict(X_full_test)
lasso_bayes_y_sub_pred = lasso_bayes_sub.predict(X_sub_test)

# Inverse transformation for predicted values
lasso_bayes_y_full_pred_inverse = np.expm1(lasso_bayes_y_full_pred)
lasso_bayes_y_sub_pred_inverse = np.expm1(lasso_bayes_y_sub_pred)

# Calculate and print evaluation metrics
mse_lasso_bayes_full = mean_squared_error(y_full_test, lasso_bayes_y_full_pred_inverse)
mse_lasso_bayes_sub = mean_squared_error(y_sub_test, lasso_bayes_y_sub_pred_inverse)

rmse_lasso_bayes_full = np.sqrt(mse_lasso_bayes_full)
rmse_lasso_bayes_sub = np.sqrt(mse_lasso_bayes_sub)

r2_lasso_bayes_full = r2_score(y_full_test, lasso_bayes_y_full_pred_inverse)
r2_lasso_bayes_sub = r2_score(y_sub_test, lasso_bayes_y_sub_pred_inverse)

lasso_bayes_full_best_params =  lasso_bayes_full.best_params_
lasso_bayes_sub_best_params =   lasso_bayes_sub.best_params_

###############################################################
# Modelling with RandomizedSearchCV for Hyperparametertuning
###############################################################

# Use RandomizedSearchCV with the defined search space
lasso_random_full = RandomizedSearchCV(lasso_full, param_distributions=lasso_random_params, n_iter=20, cv=kfolds, n_jobs=-1)
lasso_random_sub = RandomizedSearchCV(lasso_sub, param_distributions=lasso_random_params, n_iter=20, cv=kfolds, n_jobs=-1)

# Fit the models
lasso_random_full.fit(X_full_train, y_full_train)
lasso_random_sub.fit(X_sub_train, y_sub_train)

# Make predictions
lasso_random_y_full_pred = lasso_random_full.predict(X_full_test)
lasso_random_y_sub_pred = lasso_random_sub.predict(X_sub_test)

# Inverse transformation for predicted values
lasso_random_y_full_pred_inverse = np.expm1(lasso_random_y_full_pred)
lasso_random_y_sub_pred_inverse = np.expm1(lasso_random_y_sub_pred)

# Calculate and print evaluation metrics
mse_lasso_random_full = mean_squared_error(y_full_test, lasso_random_y_full_pred_inverse)
mse_lasso_random_sub = mean_squared_error(y_sub_test, lasso_random_y_sub_pred_inverse)

rmse_lasso_random_full = np.sqrt(mse_lasso_random_full)
rmse_lasso_random_sub = np.sqrt(mse_lasso_random_sub)

r2_lasso_random_full = r2_score(y_full_test, lasso_random_y_full_pred_inverse)
r2_lasso_random_sub = r2_score(y_sub_test, lasso_random_y_sub_pred_inverse)

lasso_random_full_best_params =  lasso_random_full.best_params_
lasso_random_sub_best_params =   lasso_random_sub.best_params_

###############################################################
# Modelling with GridSearchCV for Hyperparametertuning
###############################################################

# Use GridSearchCV instead of BayesSearchCV
lasso_grid_full = GridSearchCV(lasso_full, lasso_grid_params, cv=kfolds)
lasso_grid_sub = GridSearchCV(lasso_sub, lasso_grid_params, cv=kfolds)

# Fit the models
lasso_grid_full.fit(X_full_train, y_full_train)
lasso_grid_sub.fit(X_sub_train, y_sub_train)

# Make predictions
lasso_grid_y_full_pred = lasso_grid_full.predict(X_full_test)
lasso_grid_y_sub_pred = lasso_grid_sub.predict(X_sub_test)

# Inverse transformation for predicted values
lasso_grid_y_full_pred_inverse = np.expm1(lasso_grid_y_full_pred)
lasso_grid_y_sub_pred_inverse = np.expm1(lasso_grid_y_sub_pred)

# Calculate and print evaluation metrics
mse_lasso_grid_full = mean_squared_error(y_full_test, lasso_grid_y_full_pred_inverse)
mse_lasso_grid_sub = mean_squared_error(y_sub_test, lasso_grid_y_sub_pred_inverse)

rmse_lasso_grid_full = np.sqrt(mse_lasso_grid_full)
rmse_lasso_grid_sub = np.sqrt(mse_lasso_grid_sub)

r2_lasso_grid_full = r2_score(y_full_test, lasso_grid_y_full_pred_inverse)
r2_lasso_grid_sub = r2_score(y_sub_test, lasso_grid_y_sub_pred_inverse)

lasso_grid_full_best_params =  lasso_grid_full.best_params_
lasso_grid_sub_best_params =   lasso_grid_sub.best_params_

###############################################################
# Save results
###############################################################

# Create a list to store the results
results_lasso_list = []

# Append results for the full data using Bayes
results_lasso_list.append({
    'data': 'full',
    'best_alpha': lasso_bayes_full_best_params['alpha'],
    'mse': mse_lasso_bayes_full,
    'rmse': rmse_lasso_bayes_full,
    'r_square': r2_lasso_bayes_full,
    'tuning_method': 'bayes'
})

# Append results for the sub data using Bayes
results_lasso_list.append({
    'data': 'sub',
    'best_alpha': lasso_bayes_sub_best_params['alpha'],
    'mse': mse_lasso_bayes_sub,
    'rmse': rmse_lasso_bayes_sub,
    'r_square': r2_lasso_bayes_sub,
    'tuning_method': 'bayes'
})

# Append results for the full data using Random
results_lasso_list.append({
    'data': 'full',
    'best_alpha': lasso_random_full_best_params['alpha'],
    'mse': mse_lasso_random_full,
    'rmse': rmse_lasso_random_full,
    'r_square': r2_lasso_random_full,
    'tuning_method': 'random'
})

# Append results for the sub data using Random
results_lasso_list.append({
    'data': 'sub',
    'best_alpha': lasso_random_sub_best_params['alpha'],
    'mse': mse_lasso_random_sub,
    'rmse': rmse_lasso_random_sub,
    'r_square': r2_lasso_random_sub,
    'tuning_method': 'random'
})

# Append results for the full data using Grid
results_lasso_list.append({
    'data': 'full',
    'best_alpha': lasso_grid_full_best_params['alpha'],
    'mse': mse_lasso_grid_full,
    'rmse': rmse_lasso_grid_full,
    'r_square': r2_lasso_grid_full,
    'tuning_method': 'grid'
})

# Append results for the sub data using Grid
results_lasso_list.append({
    'data': 'sub',
    'best_alpha': lasso_grid_sub_best_params['alpha'],
    'mse': mse_lasso_grid_sub,
    'rmse': rmse_lasso_grid_sub,
    'r_square': r2_lasso_grid_sub,
    'tuning_method': 'grid'
})

# Convert the list of dictionaries to a DataFrame
results_lasso = pd.DataFrame(results_lasso_list)

# Reorder rows
new_order = [0, 2, 4, 1, 3, 5]
results_lasso = results_lasso.loc[new_order].reset_index(drop=True)

# Save
results_lasso.to_csv(os.path.join("./models/model_evaluation/", "results_lasso.csv"), index=False)

###############################################################
# Save best model
###############################################################
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

ridge_full = Ridge()
ridge_sub = Ridge()

kfolds = KFold(n_splits=5, shuffle=True, random_state=42)
ridge_bayes_params = {'alpha': (0.1, 1)}
ridge_random_params = {'alpha': np.logspace(-1, 0, 100)}
ridge_grid_params = {'alpha': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]} 

###############################################################
# Modelling with BayesSearchCV for Hyperparameter tuning
###############################################################

# Use BayesSearchCV with the defined search space
ridge_bayes_full = BayesSearchCV(ridge_full, search_spaces=ridge_bayes_params, n_iter=20, cv=kfolds, n_jobs=-1)
ridge_bayes_sub = BayesSearchCV(ridge_sub, search_spaces=ridge_bayes_params, n_iter=20, cv=kfolds, n_jobs=-1)

# Fit the models
ridge_bayes_full.fit(X_full_train, y_full_train)
ridge_bayes_sub.fit(X_sub_train, y_sub_train)

# Make predictions
ridge_bayes_y_full_pred = ridge_bayes_full.predict(X_full_test)
ridge_bayes_y_sub_pred = ridge_bayes_sub.predict(X_sub_test)

# Inverse transformation for predicted values
ridge_bayes_y_full_pred_inverse = np.expm1(ridge_bayes_y_full_pred)
ridge_bayes_y_sub_pred_inverse = np.expm1(ridge_bayes_y_sub_pred)

# Calculate and print evaluation metrics
mse_ridge_bayes_full = mean_squared_error(y_full_test, ridge_bayes_y_full_pred_inverse)
mse_ridge_bayes_sub = mean_squared_error(y_sub_test, ridge_bayes_y_sub_pred_inverse)

rmse_ridge_bayes_full = np.sqrt(mse_ridge_bayes_full)
rmse_ridge_bayes_sub = np.sqrt(mse_ridge_bayes_sub)

r2_ridge_bayes_full = r2_score(y_full_test, ridge_bayes_y_full_pred_inverse)
r2_ridge_bayes_sub = r2_score(y_sub_test, ridge_bayes_y_sub_pred_inverse)

ridge_bayes_full_best_params =  ridge_bayes_full.best_params_
ridge_bayes_sub_best_params =   ridge_bayes_sub.best_params_

###############################################################
# Modelling with RandomizedSearchCV for Hyperparameter tuning
###############################################################

# Use RandomizedSearchCV with the defined search space
ridge_random_full = RandomizedSearchCV(ridge_full, param_distributions=ridge_random_params, n_iter=20, cv=kfolds, n_jobs=-1)
ridge_random_sub = RandomizedSearchCV(ridge_sub, param_distributions=ridge_random_params, n_iter=20, cv=kfolds, n_jobs=-1)

# Fit the models
ridge_random_full.fit(X_full_train, y_full_train)
ridge_random_sub.fit(X_sub_train, y_sub_train)

# Make predictions
ridge_random_y_full_pred = ridge_random_full.predict(X_full_test)
ridge_random_y_sub_pred = ridge_random_sub.predict(X_sub_test)

# Inverse transformation for predicted values
ridge_random_y_full_pred_inverse = np.expm1(ridge_random_y_full_pred)
ridge_random_y_sub_pred_inverse = np.expm1(ridge_random_y_sub_pred)

# Calculate and print evaluation metrics
mse_ridge_random_full = mean_squared_error(y_full_test, ridge_random_y_full_pred_inverse)
mse_ridge_random_sub = mean_squared_error(y_sub_test, ridge_random_y_sub_pred_inverse)

rmse_ridge_random_full = np.sqrt(mse_ridge_random_full)
rmse_ridge_random_sub = np.sqrt(mse_ridge_random_sub)

r2_ridge_random_full = r2_score(y_full_test, ridge_random_y_full_pred_inverse)
r2_ridge_random_sub = r2_score(y_sub_test, ridge_random_y_sub_pred_inverse)

ridge_random_full_best_params =  ridge_random_full.best_params_
ridge_random_sub_best_params =   ridge_random_sub.best_params_

###############################################################
# Modelling with GridSearchCV for Hyperparameter tuning
###############################################################

# Use GridSearchCV instead of BayesSearchCV
ridge_grid_full = GridSearchCV(ridge_full, ridge_grid_params, cv=kfolds)
ridge_grid_sub = GridSearchCV(ridge_sub, ridge_grid_params, cv=kfolds)

# Fit the models
ridge_grid_full.fit(X_full_train, y_full_train)
ridge_grid_sub.fit(X_sub_train, y_sub_train)

# Make predictions
ridge_grid_y_full_pred = ridge_grid_full.predict(X_full_test)
ridge_grid_y_sub_pred = ridge_grid_sub.predict(X_sub_test)

# Inverse transformation for predicted values
ridge_grid_y_full_pred_inverse = np.expm1(ridge_grid_y_full_pred)
ridge_grid_y_sub_pred_inverse = np.expm1(ridge_grid_y_sub_pred)

# Calculate and print evaluation metrics
mse_ridge_grid_full = mean_squared_error(y_full_test, ridge_grid_y_full_pred_inverse)
mse_ridge_grid_sub = mean_squared_error(y_sub_test, ridge_grid_y_sub_pred_inverse)

rmse_ridge_grid_full = np.sqrt(mse_ridge_grid_full)
rmse_ridge_grid_sub = np.sqrt(mse_ridge_grid_sub)

r2_ridge_grid_full = r2_score(y_full_test, ridge_grid_y_full_pred_inverse)
r2_ridge_grid_sub = r2_score(y_sub_test, ridge_grid_y_sub_pred_inverse)

ridge_grid_full_best_params =  ridge_grid_full.best_params_
ridge_grid_sub_best_params =   ridge_grid_sub.best_params_

###############################################################
# Save results
###############################################################

# Create a list to store the results
results_ridge_list = []

# Append results for the full data using Bayes
results_ridge_list.append({
    'data': 'full',
    'best_alpha': ridge_bayes_full_best_params['alpha'],
    'mse': mse_ridge_bayes_full,
    'rmse': rmse_ridge_bayes_full,
    'r_square': r2_ridge_bayes_full,
    'tuning_method': 'bayes'
})

# Append results for the sub data using Bayes
results_ridge_list.append({
    'data': 'sub',
    'best_alpha': ridge_bayes_sub_best_params['alpha'],
    'mse': mse_ridge_bayes_sub,
    'rmse': rmse_ridge_bayes_sub,
    'r_square': r2_ridge_bayes_sub,
    'tuning_method': 'bayes'
})

# Append results for the full data using Random
results_ridge_list.append({
    'data': 'full',
    'best_alpha': ridge_random_full_best_params['alpha'],
    'mse': mse_ridge_random_full,
    'rmse': rmse_ridge_random_full,
    'r_square': r2_ridge_random_full,
    'tuning_method': 'random'
})

# Append results for the sub data using Random
results_ridge_list.append({
    'data': 'sub',
    'best_alpha': ridge_random_sub_best_params['alpha'],
    'mse': mse_ridge_random_sub,
    'rmse': rmse_ridge_random_sub,
    'r_square': r2_ridge_random_sub,
    'tuning_method': 'random'
})

# Append results for the full data using Grid
results_ridge_list.append({
    'data': 'full',
    'best_alpha': ridge_grid_full_best_params['alpha'],
    'mse': mse_ridge_grid_full,
    'rmse': rmse_ridge_grid_full,
    'r_square': r2_ridge_grid_full,
    'tuning_method': 'grid'
})

# Append results for the sub data using Grid
results_ridge_list.append({
    'data': 'sub',
    'best_alpha': ridge_grid_sub_best_params['alpha'],
    'mse': mse_ridge_grid_sub,
    'rmse': rmse_ridge_grid_sub,
    'r_square': r2_ridge_grid_sub,
    'tuning_method': 'grid'
})

# Convert the list of dictionaries to a DataFrame
results_ridge = pd.DataFrame(results_ridge_list)

# Reorder rows
new_order = [0, 2, 4, 1, 3, 5]
results_ridge = results_ridge.loc[new_order].reset_index(drop=True)

# Save
results_ridge.to_csv(os.path.join("./models/model_evaluation/", "results_ridge.csv"), index=False)

###############################################################
# Save best models
###############################################################

joblib.dump(ridge_bayes_full, 'ridge_bayes_full_model.pkl')
joblib.dump(ridge_bayes_sub, 'ridge_bayes_sub_model.pkl')
joblib.dump(ridge_random_full, 'ridge_random_full_model.pkl')
joblib.dump(ridge_random_sub, 'ridge_random_sub_model.pkl')
joblib.dump(ridge_grid_full, 'ridge_grid_full_model.pkl')
joblib.dump(ridge_grid_sub, 'ridge_grid_sub_model.pkl')
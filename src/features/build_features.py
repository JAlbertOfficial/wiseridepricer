###############################################################
# Modules import
###############################################################

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 
import csv
import os

from matplotlib.cm import viridis
from scipy import stats
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PolynomialFeatures, OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error,  r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

###############################################################
# Data import
###############################################################

path = "./data/raw/autoscout24.csv"

df_raw = pd.read_csv(path)

#--------------------------------------------------------------
# Create 2 seperate DataFrames
#--------------------------------------------------------------

df_interim_full = df_raw.copy()
top5makes = df_raw['make'].value_counts().head(5).index
df_interim_sub = df_raw[df_raw['make'].isin(top5makes)].copy()

#--------------------------------------------------------------
# Remove missing entries
#--------------------------------------------------------------

df_interim_full.dropna(inplace=True, ignore_index=True)
df_interim_sub.dropna(inplace=True, ignore_index=True)

#--------------------------------------------------------------
# Identify target, and continuous, ordinal, and features columns
#--------------------------------------------------------------

target = ['price']
continuous_features = ['mileage', 'hp', 'year']
ordinal_features = ['offerType']
nominal_features = ["make", "model", "fuel", "gear"]

#--------------------------------------------------------------
# Ordinal Encoding for ordinal features
#--------------------------------------------------------------

offer_type_order = ['Used', "Employee's car", 'Demonstration', 'Pre-registered', 'New']
ordinal_encoder = OrdinalEncoder(categories=[offer_type_order])

# Create a DataFrame with the original 'offerType' column
df_ordinal_full = pd.DataFrame(df_interim_full[ordinal_features])
df_ordinal_sub = pd.DataFrame(df_interim_sub[ordinal_features])

# Perform ordinal encoding and create a new DataFrame
df_interim_full[ordinal_features] = ordinal_encoder.fit_transform(df_interim_full[ordinal_features])
df_interim_sub[ordinal_features] = ordinal_encoder.fit_transform(df_interim_sub[ordinal_features])

#--------------------------------------------------------------
# One Hot Encoding for nominal features
#--------------------------------------------------------------

if any(col in df_interim_full.columns for col in nominal_features):
    # Create a new DataFrame with only the one-hot encoded columns
    df_one_hot_full = pd.get_dummies(df_interim_full[nominal_features], columns=nominal_features, dtype=int)

    # Drop the original nominal columns from the original DataFrame
    df_interim_full = df_interim_full.drop(columns=nominal_features)

    # Concatenate the original DataFrame without nominal columns and the new DataFrame with one-hot encoded columns
    df_interim_full = pd.concat([df_interim_full, df_one_hot_full], axis=1)

if any(col in df_interim_sub.columns for col in nominal_features):
    # Create a new DataFrame with only the one-hot encoded columns
    df_one_hot_sub = pd.get_dummies(df_interim_sub[nominal_features], columns=nominal_features, dtype=int)

    # Drop the original nominal columns from the original DataFrame
    df_interim_sub = df_interim_sub.drop(columns=nominal_features)

    # Concatenate the original DataFrame without nominal columns and the new DataFrame with one-hot encoded columns
    df_interim_sub = pd.concat([df_interim_sub, df_one_hot_sub], axis=1)

#--------------------------------------------------------------
# Save processed data
#--------------------------------------------------------------    

df_processed_full = df_interim_full.copy()
df_processed_sub = df_interim_sub.copy()    
df_processed_full.to_csv(os.path.join("./data/processed/", "df_processed_full.csv"), index=False)
df_processed_sub.to_csv(os.path.join("./data/processed/", "df_processed_sub.csv"), index=False)


#--------------------------------------------------------------
# Train Test Split
#--------------------------------------------------------------

X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(df_interim_full.drop("price",axis=1), 
                                                                        df_interim_full["price"], test_size = 0.2, random_state = 42)
X_sub_train, X_sub_test, y_sub_train, y_sub_test = train_test_split(df_interim_sub.drop("price",axis=1), 
                                                                        df_interim_sub["price"], test_size = 0.2, random_state = 42)

#--------------------------------------------------------------
# Log Transformation of target
#--------------------------------------------------------------
    
y_full_train = np.log1p(y_full_train)
y_sub_train = np.log1p(y_sub_train)

#y_full_test = np.log1p(y_full_test)
#y_sub_test = np.log1p(y_sub_test)

#--------------------------------------------------------------
# Cbrt Transformation of some features
#--------------------------------------------------------------

X_full_train[["mileage","hp"]] =  np.cbrt(X_full_train[["mileage","hp"]]) 
X_sub_train[["mileage","hp"]] =  np.cbrt(X_sub_train[["mileage","hp"]]) 

X_full_test[["mileage","hp"]] =  np.cbrt(X_full_test[["mileage","hp"]]) 
X_sub_test[["mileage","hp"]] =  np.cbrt(X_sub_test[["mileage","hp"]]) 

#--------------------------------------------------------------
# Normalize test and train date with minimum and maximum from train data to a range of 0 to 1
#--------------------------------------------------------------

normalize_columns = ["mileage", "hp", "year", "offerType"]

minmax_values_full = pd.DataFrame({
    'feature': normalize_columns,
    'min': X_full_train[normalize_columns].min(),
    'max': X_full_train[normalize_columns].max()
})

minmax_values_sub = pd.DataFrame({
    'feature': normalize_columns,
    'min': X_sub_train[normalize_columns].min(),
    'max': X_sub_train[normalize_columns].max()
})

# Normalize function
def normalize_value(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

# Normalize training features in full data
for feature in normalize_columns:
    min_value = minmax_values_full[minmax_values_full['feature'] == feature]['min'].values[0]
    max_value = minmax_values_full[minmax_values_full['feature'] == feature]['max'].values[0]
    
    X_full_train[feature] = X_full_train[feature].apply(lambda x: normalize_value(x, min_value, max_value))

# Normalize training features in subset data
for feature in normalize_columns:
    min_value = minmax_values_sub[minmax_values_sub['feature'] == feature]['min'].values[0]
    max_value = minmax_values_sub[minmax_values_sub['feature'] == feature]['max'].values[0]
    
    X_sub_train[feature] = X_sub_train[feature].apply(lambda x: normalize_value(x, min_value, max_value))

# Normalize test features in full data
for feature in normalize_columns:
    min_value = minmax_values_full[minmax_values_full['feature'] == feature]['min'].values[0]
    max_value = minmax_values_full[minmax_values_full['feature'] == feature]['max'].values[0]
    
    X_full_test[feature] = X_full_test[feature].apply(lambda x: normalize_value(x, min_value, max_value))

# Normalize test features in subset data
for feature in normalize_columns:
    min_value = minmax_values_sub[minmax_values_sub['feature'] == feature]['min'].values[0]
    max_value = minmax_values_sub[minmax_values_sub['feature'] == feature]['max'].values[0]
    
    X_sub_test[feature] = X_sub_test[feature].apply(lambda x: normalize_value(x, min_value, max_value))





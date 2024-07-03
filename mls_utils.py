import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

def split_data(df, features, target, test_size=0.3, random_state=42):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_decision_tree_model(X_train, y_train, random_state=42, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    model = DecisionTreeRegressor(random_state=random_state, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    model.fit(X_train, y_train)
    return model

def train_ridge_regression_model(X_train, y_train, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_lasso_regression_model(X_train, y_train, alpha=1.0):
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_random_forest_model(X_train, y_train, random_state=42, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    model.fit(X_train, y_train)
    return model

def train_svm_model(X_train, y_train, C=1.0, gamma='scale'):
    model = SVR(C=C, gamma=gamma)
    model.fit(X_train, y_train)
    return model

def train_knn_model(X_train, y_train, n_neighbors=5):
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def tune_hyperparameters(X_train, y_train, model_type):
    if model_type == 'decision_tree':
        param_grid = {'max_depth': range(1, 21), 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        model = DecisionTreeRegressor()
    elif model_type == 'svm':
        param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
        model = SVR()
    elif model_type == 'knn':
        param_grid = {'n_neighbors': range(1, 31)}
        model = KNeighborsRegressor()
    elif model_type == 'ridge':
        param_grid = {'alpha': [0.1, 1, 10]}
        model = Ridge()
    elif model_type == 'lasso':
        param_grid = {'alpha': [0.1, 1, 10]}
        model = Lasso()
    elif model_type == 'random_forest':
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        model = RandomForestRegressor()
    else:
        raise ValueError(f"Model type '{model_type}' is not supported for hyperparameter tuning.")

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def calculate_bias_variance(y_true, y_pred):
    bias = np.mean((y_true - np.mean(y_pred))**2)
    variance = np.mean((y_pred - np.mean(y_pred))**2)
    return bias, variance

def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    return train_mse, test_mse, train_r2, test_r2

def select_features(X_train, y_train, model=None):
    if model is None:
        model = Lasso(alpha=0.1)
    selector = SelectFromModel(model)
    selector.fit(X_train, y_train)
    return selector.get_support()
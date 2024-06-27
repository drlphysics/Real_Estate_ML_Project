import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


def split_data(df, features, target, test_size=0.2, random_state=42):
    # Split the data into X and y.  Change test size and random state as needed.
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_decision_tree_model(X_train, y_train, random_state=42, max_depth=None):
    #Train a DecisionTreeRegressor model
    regressor = DecisionTreeRegressor(random_state=random_state, max_depth=max_depth)
    regressor.fit(X_train, y_train)
    return regressor

def visualize_tree(regressor, feature_names):
    #Visualize the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(regressor, feature_names=feature_names, filled=True, rounded=True, fontsize=12)
    plt.show()

def train_linear_regression_model(X_train, y_train, alpha=0.0):
    #Train a LinearRegression model
    regressor = Ridge(alpha=alpha)
    regressor.fit(X_train, y_train)
    return regressor

def train_random_forest_model(X_train, y_train, random_state=42, n_estimators=100, max_depth=None):
    #Train a RandomForestRegressor model
    regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)
    regressor.fit(X_train, y_train)
    return regressor

def train_svm_model(X_train, y_train, kernel='linear', C=1.0, epsilon=0.1):
    #Train a SVM model
    regressor = SVR(kernel=kernel, C=C, epsilon=epsilon)
    regressor.fit(X_train, y_train)
    return regressor

def train_knn_model(X_train, y_train, n_neighbors=5):
    #Train a KNeighborsRegressor model
    regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(regressor, X_train, X_test, y_train, y_test):
    #Evaluate the trained models.
    y_pred_train = regressor.predict(X_train)
    y_pred_test = regressor.predict(X_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    return train_mse, test_mse, train_r2, test_r2

def tune_hyperparameters(X_train, y_train, model_type):
    """Tune hyperparameters for the given model type."""
    if model_type == 'decision_tree':
        param_grid = {'max_depth': [3, 5, 7, 10, None]}
        model = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5)
    elif model_type == 'svm':
        param_grid = {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1, 0.5], 'kernel': ['linear', 'rbf']}
        model = GridSearchCV(SVR(), param_grid, cv=5)
    elif model_type == 'knn':
        param_grid = {'n_neighbors': [3, 5, 7, 10]}
        model = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5)
    model.fit(X_train, y_train)
    return model.best_estimator_








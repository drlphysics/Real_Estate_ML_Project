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

def train_decision_tree_model(X_train, y_train, random_state=42):
    #Train a DecisionTreeRegressor model
    regressor = DecisionTreeRegressor(random_state=random_state)
    regressor.fit(X_train, y_train)
    return regressor

def visualize_tree(regressor, feature_names):
    #Visualize the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(regressor, feature_names=feature_names, filled=True, rounded=True, fontsize=12)
    plt.show()

def train_linear_regression_model(X_train, y_train):
    #Train a LinearRegression model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor

def train_random_forest_model(X_train, y_train, random_state=42, n_estimators=100):
    #Train a RandomForestRegressor model
    regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
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








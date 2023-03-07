## UTIL FUNCTIONS TO USE IN FEATURE SELECTION FOLDER
## -------------------------------------------------------------------##

# for data processing and manipulation
import os
import pandas as pd

# scikit-learn modules for feature engineering
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# scikit-learn modules for feature selection and model evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

## -------------------------------------------------------------------##
##1. Function to create new folder
def folder_create(folder):
    """
    Function to check if a folder exist, otherwise, create one named like indicated
    :params: 
        folder - name of the new folder 
    :return: 
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

## -------------------------------------------------------------------##
##2. Feature engineering
def FE_tranform(X):
    """
    Function to apply this feature engineering
    :params: 
        X - raw features
    :return: 
        X_tranform - features transformed
    """
    numerical_columns_selector = selector(dtype_exclude=object)
    numerical_columns = numerical_columns_selector(X)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_columns_selector = selector(dtype_include=object)
    categorical_columns = categorical_columns_selector(X)
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)])

    X_tranform = preprocessor.fit_transform(X)
    return X_tranform


## -------------------------------------------------------------------##
##3. Function to calculate the metrics for our model
def calculate_metrics(model, X_test, Y_test):
    """
    Function to get model evaluation metrics on the test set
    :params:
        model - RandomForest model which automatically tranform features
        X_test - test partition of input features
        Y_test - test partition of output feature
    :return:
        mse - mean squared error
        rmse - root mean squared error
        mae - mean absolute error
    """
    
    # Get model predictions
    y_predict_r = model.predict(X_test)
    
    # Calculate evaluation metrics for assesing performance of the model.
    mse = mean_squared_error(Y_test, y_predict_r)
    rmse = mean_squared_error(Y_test, y_predict_r, squared=False)
    mae = mean_absolute_error(Y_test, y_predict_r)
    
    return mse, rmse, mae

## -------------------------------------------------------------------##
##4. Function to fit the model
def fit_model(X, Y):
    """
   Function to fit RandomForestRegressor
    :params:
        X - input features
        Y - output feature
    :return:
        model - fitted RandomForestRegressor model
    """

    model = RandomForestRegressor(random_state=20)

    # Train the model
    model.fit(X, Y)
    
    return model


## -------------------------------------------------------------------##
##5. Function to get metrics
def train_and_get_metrics(X, Y):
    """
    Function to train a RandomForestRegressor and get evaluation metrics
    :params:
        X - input features
        Y - output feature
    :return:
        mse - mean squared error
        rmse - root mean squared error
        mae - mean absolute error
    """
    
    # Split train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1995)

    # Call the fit model function to train the model
    model = fit_model(X_train, Y_train)

    # Make predictions on test dataset and calculate metrics
    mse, rmse, mae = calculate_metrics(model, X_test, Y_test)

    return mse, rmse, mae


## -------------------------------------------------------------------##
##6. Function to evaluate predictions
def evaluate_model_on_features(X, Y):
    """
    Function to train model and display evaluation metrics
    :params:
        X - input features
        Y - output feature
    :return:
        display_df - dataframe with number of features and performance metrics achieved for test partition
    """
    
    # Train the model, predict values and get metrics
    mse, rmse, mae = train_and_get_metrics(X, Y)

    # Construct a dataframe to display metrics.
    display_df = pd.DataFrame([[mse, rmse, mae, X.shape[1]]], columns=["MSE", "RMSE", "MAE", 'Feature Count'])
    
    return display_df

## -------------------------------------------------------------------##
##7. Function to use Recursive Feature Elimination for Wrapped method
def run_rfe(X, Y):
    """
    Function to train model and display evaluation metrics
    :params:
        X - input features
        Y - output feature
    :return:
        rfe - Recursive Feature Elimination filter of features
    """
    # Split train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1995)
    
    # Normalize all features of the train.
    X_train_scaled = FE_tranform(X_train)

    # Wrap RFE around the model
    model = RandomForestRegressor(random_state=20)
    rfe = RFE(estimator=model, n_features_to_select=20, step=1)
    
    # Fit RFE
    rfe = rfe.fit(X_train_scaled, Y_train)
    feature_filter = rfe.get_support()
    
    return feature_filter

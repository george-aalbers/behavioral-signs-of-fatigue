# Required packages
import os
import time
import argparse
import pandas as pd
import numpy as np
import optuna
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

# Start the timer
start_time = time.perf_counter()

# Define functions
def read_data(participant_id):
    
    # Read the full dataset
    data = pd.read_csv("data.csv", index_col = 0)
    
    # Select data for one participant
    data = data[data.id == participant_id]
    
    return data

def get_id(df):
    return df.id.unique()[0]

def write_to_file(df, filename):
    
    # If the file does not exist, create new file
    if filename not in os.listdir():
        
        df.to_csv(filename)
        
    # If the file does exist, add a line to existing .csv
    else:
        
        df.to_csv(filename, mode = 'a', header = None)

def time_split(data, split):
    
    # Splitting the data for month 1
    if split == 1:

        # Get first month
        data = data.iloc[:150,:]
        
        # Get shape
        n1 = int(data.shape[0]/2)
        n2 = int(data.shape[0]/4) * 3
        
        # Select data
        train_data = data.iloc[:n1,:]
        val_data = data.iloc[n1:n2,:]
        test_data = data.iloc[n2:,:]
    
    # Splitting the data for month 2
    else:
        
        # Get second month
        data = data.iloc[150:300,:]
        
        # Get shape
        n1 = int(data.shape[0]/2)
        n2 = int(data.shape[0]/4) * 3
        
        # Select data
        train_data = data.iloc[:n1,:]
        val_data = data.iloc[n1:n2,:]
        test_data = data.iloc[n2:,:]
        
    return train_data, val_data, test_data

def X_y_split(data, target, features):

    if (features == 'time'):
        
        features = ['Hour of day', 
                    'Day of week']
        
    elif (features == 'app'):
        
        features = ['Browser (duration before survey)',
                    'Video (duration before survey)',
                    'Game (duration before survey)',
                    'Messenger (duration before survey)', 
                    'Social_Networks (duration before survey)'] 
    
    elif (features == 'time + app'):

        features = ['Hour of day', 
                    'Day of week', 
                    'Browser (duration before survey)',
                    'Video (duration before survey)',
                    'Game (duration before survey)',
                    'Messenger (duration before survey)', 
                    'Social_Networks (duration before survey)'] 
        
    X = data[features]
    y = data[target]
             
    return X, y

def scale_data(X_train, y_train, X_val, y_val, X_test, y_test):
    
    # Save the feature names
    features = X_train.columns
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit scaler to train features
    scaler.fit(X_train)
    
    # Scale the train features
    X_train = pd.DataFrame(scaler.transform(X_train))
    
    # Scale the validation features based on the train features
    X_val = pd.DataFrame(scaler.transform(X_val))
    
    # Scale the test features based on the train features
    X_test = pd.DataFrame(scaler.transform(X_test))
    
    # Give new objects proper column names
    X_train.columns = features
    X_val.columns = features
    X_test.columns = features

    # Initialize scaler for targets
    scaler = StandardScaler()
    
    # Fit the scaler to targets
    scaler.fit(np.asarray(y_train).reshape(-1,1))
    
    # Scale the train targets
    y_train = pd.DataFrame(scaler.transform(np.asarray(y_train).reshape(-1,1)))    
    
    # Scale the validation targets
    y_val = pd.DataFrame(scaler.transform(np.asarray(y_val).reshape(-1,1)))
    
    # Scale the test targets
    y_test = pd.DataFrame(scaler.transform(np.asarray(y_test).reshape(-1,1)))
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def prepare_data_single_setting(df, id_var, target, features, split):
    
    # Split the data into train, validation, test
    train, validation, test = time_split(df, split)

    # Split the data into X and y
    X_train, y_train = X_y_split(train, target, features)
    X_val, y_val     = X_y_split(validation, target, features)
    X_test, y_test   = X_y_split(test, target, features)

    # Scale the data
    X_train, y_train, X_val, y_val, X_test, y_test = scale_data(X_train, y_train, X_val, y_val, X_test, y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

def get_hyperparameters(df, id_var, split, target, features, model_type):
    
    if model_type == 'ebr':

        # Select participant
        df = df[df.iloc[:,0] == id_var]

        # Select split
        df = df[df.split == split]

        # Select features
        df = df[df.features == features]

        # Select target
        df = df[df.target == target]

        # Create dictionary of hyperparametrs
        names = df.iloc[:,2:-4].columns.tolist()
        values = df.iloc[:,2:-4].values.tolist()[0]
        df = dict(zip(names, values))
    
    elif model_type == 'ridge':
        
        # Select participant
        df = df[df.iloc[:,0] == id_var]

        # Select split
        df = df[df.split == split]

        # Select features
        df = df[df.features == features]

        # Select target
        df = df[df.target == target]
        
        # Create dictionary of hyperparametrs
        df = {'alpha':df.alpha.values.tolist()[0]}
    
    return df

def build_model(params, model_type):
 
    if model_type == 'ebr':

        model = ExplainableBoostingRegressor(max_bins = params['max_bins'],
                                             max_interaction_bins = params['max_interaction_bins'],
                                             binning = params['binning'],
                                             interactions = params['interactions'],
                                             outer_bags = params['outer_bags'],
                                             inner_bags = params['inner_bags'],
                                             min_samples_leaf = params['min_samples_leaf'],
                                             max_leaves = params['max_leaves'])
        
    elif model_type == 'ridge':
        
        model = sklearn.linear_model.Ridge(alpha=params['alpha'])
        
    return model

def train_model(X, y, model):
        
    model.fit(X, np.ravel(y))   
    
    return model

def make_predictions(X, y, model):
    return model.predict(X)

def make_predictions_df(y_true, y_pred):

    # Attach to y_true
    df = pd.concat([pd.Series(y_pred.squeeze()), y_true.reset_index()], axis = 1)
    
    # Rename columns
    df.columns = ["y_pred", "id", "y_true"]
    
    # Reorder columns
    df = df[["id", "y_pred", "y_true"]]
    
    return df

def get_predictions(X, y, model): 
    
    # Make predictions
    y_pred = make_predictions(X, y, model)
    
    # Make neat dataframe
    df = make_predictions_df(y, y_pred)
    
    return df.iloc[:,1:]

def calculate_r2(df):
    
    metric = pd.Series(r2_score(df.y_true, df.y_pred))
    
    return metric[0]
    
def get_params(model, features, target, split, id_var):
    
    names = model.feature_names_in_.tolist()
    coeff = model.coef_.tolist()
    
    df = pd.DataFrame(dict(zip(names, coeff)), index = [0])
    
    df['id']       = id_var
    df['features'] = features
    df['target']   = target
    df['split']    = split
        
    return df

# Go through best three participants
for pp in [['User #24035', 2], ['User #15208', 2], ['User #23924', 1]]:
    id_var = pp[0]
    split = pp[1]

    # Get data
    df = read_data(id_var)

    # Get optimized hyperparameters
    hyperparameter_df = pd.read_csv('best-trial.csv')

    # Select hyperparameters
    params = get_hyperparameters(hyperparameter_df, id_var, split, 'fatigue', 'app', model_type = 'ridge')

    # Build model with these hyperparameters
    model = build_model(params, model_type = 'ridge')

    # Get data for training and evaluating model
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_single_setting(df, id_var, 'fatigue', 'app', split)

    # Train the model
    model = train_model(X_train, y_train, model)

    # Make predictions
    predictions_train = get_predictions(X_train, y_train, model)
    predictions_val = get_predictions(X_val, y_val, model)
    predictions_test = get_predictions(X_test, y_test, model)

    sns.regplot(x = 'y_true', y = 'y_pred', data = predictions_train, marker="+", color = 'black')
    sns.despine()
    plt.xlim(predictions_train.min().min(), predictions_train.max().max())
    plt.ylim(predictions_train.min().min(), predictions_train.max().max())
    plt.xlabel('Self-reported fatigue (Tiredn-ESS score)')
    plt.ylabel('Predicted fatigue')
    plt.title('Train set')
    plt.savefig('predictions-train-' + id_var + '-app.png')
    plt.clf()

    sns.regplot(x = 'y_true', y = 'y_pred', data = predictions_val, marker="+", color = 'black')
    sns.despine()
    plt.xlim(predictions_val.min().min(), predictions_val.max().max())
    plt.ylim(predictions_val.min().min(), predictions_val.max().max())
    plt.xlabel('Self-reported fatigue (Tiredn-ESS score)')
    plt.ylabel('Predicted fatigue')
    plt.title('Validation set')
    plt.savefig('predictions-val-' + id_var + '-app.png')
    plt.clf()

    sns.regplot(x = 'y_true', y = 'y_pred', data = predictions_test, marker="+", color = 'black')
    sns.despine()
    plt.xlim(predictions_test.min().min(), predictions_test.max().max())
    plt.ylim(predictions_test.min().min(), predictions_test.max().max())
    plt.xlabel('Self-reported fatigue (Tiredn-ESS score)')
    plt.ylabel('Predicted fatigue')
    plt.title('Test set')
    plt.savefig('predictions-test-' + id_var + '-app.png')
    plt.clf()
    
# Go through best three participants
for pp in [['User #24035', 2], ['User #15208', 2], ['User #23924', 1]]:
    id_var = pp[0]
    split = pp[1]

    # Get data
    df = read_data(id_var)

    # Get optimized hyperparameters
    hyperparameter_df = pd.read_csv('best-trial.csv')

    # Select hyperparameters
    params = get_hyperparameters(hyperparameter_df, id_var, split, 'fatigue', 'time', model_type = 'ridge')

    # Build model with these hyperparameters
    model = build_model(params, model_type = 'ridge')

    # Get data for training and evaluating model
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_single_setting(df, id_var, 'fatigue', 'time', split)

    # Train the model
    model = train_model(X_train, y_train, model)

    # Make predictions
    predictions_train = get_predictions(X_train, y_train, model)
    predictions_val = get_predictions(X_val, y_val, model)
    predictions_test = get_predictions(X_test, y_test, model)

    sns.regplot(x = 'y_true', y = 'y_pred', data = predictions_train, marker="+", color = 'black')
    sns.despine()
    plt.xlim(predictions_train.min().min(), predictions_train.max().max())
    plt.ylim(predictions_train.min().min(), predictions_train.max().max())
    plt.xlabel('Self-reported fatigue (Tiredn-ESS score)')
    plt.ylabel('Predicted fatigue')
    plt.title('Train set')
    plt.savefig('predictions-train-' + id_var + '-time.png')
    plt.clf()

    sns.regplot(x = 'y_true', y = 'y_pred', data = predictions_val, marker="+", color = 'black')
    sns.despine()
    plt.xlim(predictions_val.min().min(), predictions_val.max().max())
    plt.ylim(predictions_val.min().min(), predictions_val.max().max())
    plt.xlabel('Self-reported fatigue (Tiredn-ESS score)')
    plt.ylabel('Predicted fatigue')
    plt.title('Validation set')
    plt.savefig('predictions-val-' + id_var + '-time.png')
    plt.clf()

    sns.regplot(x = 'y_true', y = 'y_pred', data = predictions_test, marker="+", color = 'black')
    sns.despine()
    plt.xlim(predictions_test.min().min(), predictions_test.max().max())
    plt.ylim(predictions_test.min().min(), predictions_test.max().max())
    plt.xlabel('Self-reported fatigue (Tiredn-ESS score)')
    plt.ylabel('Predicted fatigue')
    plt.title('Test set')
    plt.savefig('predictions-test-' + id_var + '-time.png')
    plt.clf()
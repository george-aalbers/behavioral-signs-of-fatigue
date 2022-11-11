# Required packages
import os
import time
import argparse
import pandas as pd
import numpy as np
import optuna
import sklearn
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
def read_data(participant_number):
    
    # Read the full dataset
    data = pd.read_csv("data.csv", index_col = 0)
    
    # Get a list of participants
    participant_ids = data.id.unique().tolist()
    
    # Select data for one participant
    data = data[data.id == participant_ids[participant_number]]
    
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
    
    return df

def calculate_r2(df):
    
    metric = pd.Series(r2_score(df.y_true, df.y_pred))
    
    return metric[0]

# 1. Define an objective function to be maximized.
def objective(trial):

    # 2. Suggest values for the hyperparameters using a trial object.
    model_type = trial.suggest_categorical('regression', ['ridge'])
    
    if model_type == 'elasticNet':
        
         alpha = trial.suggest_float('alpha', 1e-10, 1e10, log=True)
         l1_ratio = trial.suggest_float('l1_ratio', 1e-10, 1, log=True) 
         model = sklearn.linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    
    elif model_type == 'lasso':
        
        alpha = trial.suggest_float('alpha', 1e-10, 1e10, log=True)
        model = sklearn.linear_model.Lasso(alpha=alpha)
    
    elif model_type == 'ridge':
        
        alpha = trial.suggest_float('alpha', 1e-10, 1e10, log=True)
        model = sklearn.linear_model.Ridge(alpha=alpha)
    
    elif model_type == 'huber':
        
        alpha = trial.suggest_float('alpha', 1e-10, 1, log=True)
        epsilon = trial.suggest_float('epsilon', 1, 10, log=True) 
        model = sklearn.linear_model.HuberRegressor(alpha=alpha, epsilon=epsilon)
    
    elif model_type == 'rf':
        
        n_estimators = trial.suggest_int('n_estimators', 1, 1e10, log=True)
        max_depth = trial.suggest_int('max_depth', 1, 1e10, log=True)
        min_samples_split = trial.suggest_float('min_samples_split', 1e-10, 1, log=True)
        min_samples_leaf = trial.suggest_float('min_samples_leaf', 1e-10, 1, log=True)
        min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 1e-10, 1, log=True)
        max_features = trial.suggest_float('max_features', 1e-10, 1, log=True)
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 1, 100, log=True)
        min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 1e-10, 1, log=True)
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=n_estimators, 
                                                       max_depth=max_depth, 
                                                       min_samples_split=min_samples_split,
                                                       min_samples_leaf=min_samples_leaf, 
                                                       min_weight_fraction_leaf = min_weight_fraction_leaf,
                                                       max_features=max_features, 
                                                       max_leaf_nodes=max_leaf_nodes,
                                                       min_impurity_decrease=min_impurity_decrease)
    
    elif model_type == 'ebr':
        
        max_bins = trial.suggest_int('max_bins', 2, 256, log=True)
        max_interaction_bins = trial.suggest_int('max_interaction_bins', 2, 32, log=True)
        binning= trial.suggest_categorical('binning',['uniform', 'quantile', 'quantile_humanized'])
        interactions = trial.suggest_int('interactions', 2, 5, log=True)
        outer_bags = trial.suggest_int('outer_bags', 1, 8, log=True)
        inner_bags = trial.suggest_int('inner_bags', 1, 8, log=True)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 100, log=True)
        max_leaves = trial.suggest_int('max_leaves', 2, 100, log=True)
        model = ExplainableBoostingRegressor(max_bins = max_bins,
                                             max_interaction_bins = max_interaction_bins,
                                             binning = binning,
                                             interactions = interactions,
                                             outer_bags = outer_bags,
                                             inner_bags = inner_bags,
                                             min_samples_leaf = min_samples_leaf,
                                             max_leaves = max_leaves)
        
              
    # 3. Fit the model 
    model = train_model(X_train, y_train, model)
    predictions = get_predictions(X_val, y_val, model)
    accuracy = calculate_r2(predictions)
        
    return accuracy

def add_r2(df, study):
    
    # Calculate MAE for model and baseline model on validation set
    df['r2_validation'] = study.best_trial.value
    
    return df

def add_metaparams(df, features, target, split):
    
    # Add the features
    df['features'] = features
    
    # Add the target
    df['target'] = target
    
    # Add the split
    df['split'] = split
    
    return df
    
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
    
def write_results_to_file(id_var, study, features, target, split):
    
    # Get the best hyperparameters
    results = pd.DataFrame(study.best_trial.params, index = [id_var])
    
    # Add metaparameters
    results = add_metaparams(results, features, target, split)
    
    # Add MAE
    results = add_r2(results, study)
    
    # Write results to file
    write_to_file(results, 'best-trial.csv')
    
# Create argparse arguments
parser = argparse.ArgumentParser(description='Loop through participant numbers.')
parser.add_argument('participant_i', metavar='', type=int, nargs='+', help='the initial parameters')
args = parser.parse_args()

# Get data from one participant
df = read_data(args.participant_i[0])

# Get id_var
id_var = get_id(df)

# Do analysis

# Loop through splits
for split in range(1,3,1):
    # Loop through targets
    for target in ['fatigue']:
        # Loop through feature sets
        for features in ['time', 'app', 'time + app']: 
            
            try:

                # Get data for training and evaluating model
                X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_single_setting(df, id_var, target, features, split)

                # Optimize hyperparameters
                study = optuna.create_study(direction='maximize') 

                # Maximize r-squared on the validation set (100 trials)
                study.optimize(objective, n_trials=100)

                # Write hyperparameters to file
                write_results_to_file(id_var, study, features, target, split)
                
            except:
                
                pass
            
# Calculate how long it took to train on this individual
end_time = time.perf_counter()
duration = pd.DataFrame({'id':args.participant_i[0], 'duration':end_time - start_time}, index = ['duration'])
write_to_file(duration, 'duration.csv')
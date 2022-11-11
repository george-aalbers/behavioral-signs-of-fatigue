# Functions for preprocessing of the data

# Imports
import os
import pandas as pd
import numpy as np
from feature_extraction_tools import asof_merge, calculate_duration, audit_asof_merge, define_time_windows, extract_smartphone_application_usage_features, merge_before_after_features
from temporal_features import temporal_features

def feature_extraction(esm_data, log_data, metaparameters):    
    '''
    Description
    ---
    This function does feature extraction from smartphone application usage data.

    Input
    ---
    param metaparameters: a vector with the feature extraction specification of one experiment.

    Output
    ---
    The output of this script is a set of features for one specific experiment.
    ---
    '''
    
    # Extract smartphone application usage features
    df_before, df_after = extract_smartphone_application_usage_features(esm_data, log_data, metaparameters)
    
    # Merge features before and after survey
    df = merge_before_after_features(df_before, df_after, metaparameters)

    # Extract and add temporal features
    df = temporal_features(df)
    
    # Write data to file
    df.to_csv("data.csv")

def drop_superfluous_columns():    
    # Metaparameters
    metaparameters = pd.read_json("metaparameters.json").iloc[0,:]
    
    # Read data
    df = pd.read_csv("data.csv", index_col = 0)
    
    # Get features, targets, and id variable names
    variables = metaparameters["features"]
    variables.extend(metaparameters["targets"])
    variables.append("Response Time_ESM_day")
    variables.append(metaparameters["id_variable"])
    
    # Reverse variable names
    variables.reverse()
    
    # Select those variables
    df = df[variables]
    
    # Write to file
    df.to_csv("data.csv") 
    
def read_data():
    columns = ['id',  
               'How happy do you feel right now?',
               'I feel relaxed',
               'I feel stressed (tense restless nervous or anxious)',
               'I wasted time by doing other things than what I had intended to do.',
               'I delayed before starting on work I have to do',
               'I thought: "I\'ll do it later."',
               'I have enough energy', 
               'I feel a desire to do things',
               'I can concentrate well',
               'Social_Networks (duration before survey)', 
               'Messenger (duration before survey)', 
               'Game (duration before survey)', 
               'Video (duration before survey)', 
               'Browser (duration before survey)',                
               'Hour of day', 
               'Day of week', 
               'Day of month',
               'Month']
    data = pd.read_csv("data.csv", usecols = columns)
    return data

def aggregate_data(data):
    data["fatigue"] = data[['I have enough energy', 'I feel a desire to do things', 'I can concentrate well',]].mean(axis = 1)
    data["stress"] = data[['I feel relaxed', 'I feel stressed (tense restless nervous or anxious)']].mean(axis = 1)
    data["procrastination"] = data[['I wasted time by doing other things than what I had intended to do.', 
                                    'I delayed before starting on work I have to do',
                                    'I thought: "I\'ll do it later."']].mean(axis = 1)
    return data

def sort_data(data):
    return data.sort_values(by=["id", "Month", "Day of month", "Hour of day"])

def select_data(data):
    variables = ['id',
                 'fatigue',
                 'Messenger (duration before survey)',                  
                 'Social_Networks (duration before survey)', 
                 'Game (duration before survey)', 
                 'Video (duration before survey)', 
                 'Browser (duration before survey)',                  
                 'Hour of day',
                 'Day of week']
    return data[variables]

def finalize_data():
    data = read_data()
    data.to_csv('data-not-aggregated.csv')
    data = aggregate_data(data)
    data = sort_data(data)
    data = select_data(data)
    data.to_csv('data.csv')
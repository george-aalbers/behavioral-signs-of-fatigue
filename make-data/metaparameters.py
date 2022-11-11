'''
Description
---
This function generates a .json file containing instructions for the machine learning pipeline. 

'''

import pandas as pd
import numpy as np
import os

def metaparameters():
    
    # Root folder for the directory structure
    root_folder = os.getcwd()     

    # Number of experiments
    n_experiments = 3
    
    # All required columns of the self-report data
    self_report_columns = {'id', 
                           'Response Time_ESM_day', 
                           'I feel rushed',
                           'I feel relaxed', 
                           'I feel stressed (tense restless nervous or anxious)',
                           'I have enough energy', 
                           'I feel a desire to do things',
                           'I can concentrate well',
                           'I delayed before starting on work I have to do',
                           'I wasted time by doing other things than what I had intended to do.',
                           'I thought: "I\'ll do it later."', 
                           'How happy do you feel right now?'}
 
    # Smartphone application categories to select in feature extraction
    categories = {'Calling', 
                  'Camera', 
                  'Dating', 
                  'E_Mail', 
                  'Game', 
                  'Video', 
                  'Tracker', 
                  'Social_Networks', 
                  'Music & Audio', 
                  'Exercise', 
                  'Work', 
                  'Food & Drink', 
                  'Gallery', 
                  'Productivity', 
                  'Browser', 
                  'Messenger', 
                  'Transportation_Shared',
                  'Weather'}
    
    # Feature names
    features = {# Smartphone application usage features 
                'Messenger (duration before survey)',
                'Social_Networks (duration before survey)',
                'Video (duration before survey)',
                'Browser (duration before survey)',
                'Game (duration before survey)',
                'Messenger (duration after survey)',
                'Social_Networks (duration after survey)',
                'Dating (duration after survey)',
                'Calling (duration after survey)',
                'E_Mail (duration after survey)',
                        
                # Temporal features
                "Hour of day", 
                "Day of week", 
                "Day of month",
                "Month"}
    
    # Targets
    targets = {'I feel rushed',
               'I feel relaxed', 
               'I feel stressed (tense restless nervous or anxious)',
               'I have enough energy', 
               'I feel a desire to do things',
               'I can concentrate well',
               'I delayed before starting on work I have to do',
               'I wasted time by doing other things than what I had intended to do.',
               'I thought: "I\'ll do it later."', 
               'How happy do you feel right now?'} 
    
    # Dataframe containing instructions for the study
    metaparameters = pd.DataFrame({"esm_data_path":             np.repeat('/home/haalbers/dissertation/experience-sampling-clean.csv', n_experiments),
                                     "log_data_path":             np.repeat("/home/haalbers/dissertation/mobiledna-categorized.csv", n_experiments),
                                     "sleep_features_path":       np.repeat("/home/haalbers/dissertation/sleep-features.csv", n_experiments),
                                     "baseline_path":             np.repeat(root_folder + "/baseline/baseline_performance.csv", n_experiments),
                                     "data_output_path":          (root_folder + "/experiment-" + pd.Series(range(1, n_experiments + 1, 1)).astype(str) + "/" + "data/").values,
                                     "model_output_path":         (root_folder + "/experiment-" + pd.Series(range(1, n_experiments + 1, 1)).astype(str) + "/" + "models/").values,
                                     "results_output_path":       (root_folder + "/experiment-" + pd.Series(range(1, n_experiments + 1, 1)).astype(str) + "/" + "results/").values,
                                     "explanations_output_path":  (root_folder + "/experiment-" + pd.Series(range(1, n_experiments + 1, 1)).astype(str) + "/" + "explanations/").values,
                                     "visualizations_output_path":(root_folder + "/experiment-" + pd.Series(range(1, n_experiments + 1, 1)).astype(str) + "/" + "visualizations/").values,
                                     "data_samples_output_path":  (root_folder + "/experiment-" + pd.Series(range(1, n_experiments + 1, 1)).astype(str) + "/" + "samples/").values,
                                     "markdown_path":             np.repeat(root_folder + "/markdown/", n_experiments),
                                     "id_variable":               np.repeat("id", n_experiments),
                                     "categories":                np.tile(categories, n_experiments),
                                     "features":                  np.tile(features, n_experiments),
                                     "targets":                   np.tile(targets, n_experiments),
                                     "self_report_columns":       np.tile(self_report_columns, n_experiments),
                                     "experiment":                range(1, n_experiments + 1),
                                     "experiment_type":           ["idiographic", "idiographic", "idiographic"],
                                     "window_size":               np.repeat(60, n_experiments),
                                     "prediction_task":           np.repeat("regression", n_experiments),
                                     "cross_validation_type":     ["random","random","random"],
                                     "outer_loop_cv_k_folds":     np.repeat(5, n_experiments),
                                     "inner_loop_cv_k_folds":     np.repeat(5, n_experiments),
                                     "time_series_k_splits":      np.repeat(1, n_experiments),
                                     "time_series_test_size":     np.repeat(0.2, n_experiments),
                                     "n_jobs":                    np.repeat(124, n_experiments)}, 
                                    index = np.arange(n_experiments))
    
    # Write this dataframe to .json
    metaparameters.to_json("metaparameters.json")
    
metaparameters()
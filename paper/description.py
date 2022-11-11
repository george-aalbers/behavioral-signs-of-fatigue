import pandas as pd

def get_description():

    # Read data
    mdna = pd.read_csv("/home/haalbers/dissertation/mobiledna-clean.csv", usecols = ["id", "startTime"], low_memory=False)
    baseline = pd.read_csv("/home/haalbers/dissertation/baseline-longitudinal-clean.csv", index_col = 0, low_memory=False)
    df = pd.read_csv("data.csv", low_memory=False)
        
    # Add date so we can count days person has been in the study
    mdna["date"] = pd.to_datetime(mdna.startTime).dt.date 

    # Select included participants
    baseline = baseline[baseline.id.isin(df.id.unique().tolist())]
    
    # Get general descriptives
    n_participants = df.id.nunique()
    n_observations = df.shape[0]
    median_compliance = df.id.value_counts().median()
    std_compliance = df.id.value_counts().std()
    hours_of_logging = ( 24 * mdna.groupby('id').date.nunique().median() )    
    baseline = baseline.groupby('id').mean().reset_index()
    percentage_female = baseline.sex.value_counts().max()/n_participants * 100
    median_age = baseline.age.median()
    std_age = baseline.age.std()
    
    # Generate table with in-text values
    in_text_values = pd.DataFrame({"variable_name" : ["n_participants", "n_observations", "median_compliance", "std_compliance", "hours_of_logging",
                                                      "median_age", "std_age","percentage_female"], 
                                   "value" : [n_participants, n_observations, median_compliance, std_compliance, hours_of_logging, median_age,
                                              std_age, percentage_female]})
    
    # Write to file
    in_text_values.to_csv("description.csv")
    
get_description()
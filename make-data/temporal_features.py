# Define function for getting temporal features
def temporal_features(df):
    
    # Import pandas
    import pandas as pd
    
    # Convert to datetime object
    df["Response Time_ESM_day"] = pd.to_datetime(df["Response Time_ESM_day"].str[:18])
    
    # Extract temporal features
    df["Hour of day"] = df["Response Time_ESM_day"].dt.hour + df["Response Time_ESM_day"].dt.minute/60
    df["Day of week"] = df["Response Time_ESM_day"].dt.dayofweek
    df["Day of week"] = df["Day of week"].replace({0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:1})
    df["Day of month"] = df["Response Time_ESM_day"].dt.day
    df["Month"] = df["Response Time_ESM_day"].dt.month
    df["date"] = df["Response Time_ESM_day"].dt.date
    
    # Return df with temporal features
    return df
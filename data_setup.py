"""
Contains functions to imports the data and spilts it into train andtest splits
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def import_train_test(data_dir: str,
                      test_size: float = 0.2):
    """
    This takes a path to the data and converts it into a pandas Dataframe object
    then splits it into train and test splits according to the train_size also preprocesses
    the data according to this project

    Args:
        data_dir (string): Path to a csv file with data
        test_size (float): Size of the test split (e.g. 0.2) by Defualt this would be 0.2
    Returns:
        train_data, test_data, train_label, test_label

    """
    # Load the Data
    df = pd.read_csv(data_dir)

    # Remove Useless columns and rename colums for better understanding
    df = df.drop('Date_Time', axis=1)
    df = df.rename(columns={'Usage_kWh':'Electricity_Used_kWh',
                            'Lagging_Current_Reactive.Power_kVarh': 'Lagging_Current-Power_kVarh',
                            'Leading_Current_Reactive_Power_kVarh': 'Leading_Current-Power_kVarh',})

    # Incode text data into numbers
    encoder = LabelEncoder()
    cols_encode = ['WeekStatus', 'Day_Of_Week', 'Load_Type']
    for col in cols_encode:
        df[col] = encoder.fit_transform(df[col])
    
    df_data = df.drop(columns=['Load_Type'])
    df_target = df['Load_Type']
    
    # Split into train and test
    X_train_data, X_test_data, y_train_label, y_test_label = train_test_split(df_data, df_target, 
                                                                              test_size=test_size, 
                                                                              shuffle=True, 
                                                                              random_state=18)
    
    return X_train_data, X_test_data, y_train_label, y_test_label
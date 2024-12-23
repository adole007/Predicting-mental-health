import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path):
    """Load and perform initial data analysis"""
    data = pd.read_csv(file_path)
    print(data.head())
    print(data.describe())
    print(data.info())
    return data

def preprocess_data(data, for_training=True):
    """
    Preprocess the data for model training or prediction
    """
    # Create a copy of the data to avoid modifying the original
    processed_data = data.copy()
    
    # Define the feature columns in the correct order
    feature_columns = ['Age', 'Marital Status', 'Education Level', 'Number of Children',
                      'Smoking Status', 'Physical Activity Level', 'Employment Status',
                      'Income', 'Alcohol Consumption', 'Dietary Habits', 'Sleep Patterns',
                      'History of Substance Abuse', 'Family History of Depression',
                      'Chronic Medical Conditions']
    
    # Basic preprocessing steps
    # Convert categorical variables to numeric
    categorical_columns = ['Marital Status', 'Education Level', 'Employment Status',
                         'Smoking Status', 'Physical Activity Level', 'Alcohol Consumption',
                         'Dietary Habits', 'Sleep Patterns', 'History of Substance Abuse',
                         'Family History of Depression', 'Chronic Medical Conditions']
    
    for column in categorical_columns:
        if column in processed_data.columns:
            # Convert to category and then to numeric codes
            processed_data[column] = processed_data[column].astype('category').cat.codes
    
    # Ensure all numeric columns are float
    numeric_columns = ['Age', 'Number of Children', 'Income']
    for column in numeric_columns:
        if column in processed_data.columns:
            processed_data[column] = processed_data[column].astype(float)
    
    # Select only the feature columns in the correct order
    processed_data = processed_data[feature_columns]
    
    # Rename columns to match model expectations
    processed_data.columns = [f'f{i}' for i in range(len(feature_columns))]
    
    return processed_data
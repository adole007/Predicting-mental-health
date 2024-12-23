from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib
import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

def prepare_data(data):
    """Prepare data for model training"""
    # Create a copy to avoid modifying the original data
    data = data.copy()
    
    # Drop the Name column if it exists
    if 'Name' in data.columns:
        data = data.drop('Name', axis=1)
    
    # Define categorical columns (all non-numeric columns except the target)
    categorical_columns = [
        'Marital Status', 
        'Education Level', 
        'Smoking Status',
        'Physical Activity Level',
        'Employment Status',
        'Alcohol Consumption',
        'Dietary Habits',
        'Sleep Patterns',
        'History of Substance Abuse',
        'Family History of Depression',
        'Chronic Medical Conditions'
    ]
    
    # Define numeric columns
    numeric_columns = ['Age', 'Number of Children', 'Income']
    
    # Convert categorical columns to numeric using label encoding
    for column in categorical_columns:
        if column in data.columns:
            data[column] = pd.Categorical(data[column]).codes
    
    # Ensure numeric columns are float type
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
            # Fill any missing values with mean
            data[column] = data[column].fillna(data[column].mean())
    
    # Check if the target column exists with exact name
    target_column = 'History of Mental Illness'
    
    # Try different possible variations of the column name
    if target_column not in data.columns:
        possible_names = [
            'History_of_Mental_Illness',
            'Mental_Illness_History',
            'Mental Illness History',
            'mental_illness'
        ]
        for name in possible_names:
            if name in data.columns:
                target_column = name
                break
        else:
            print("Available columns:", data.columns.tolist())
            raise KeyError(f"Target column '{target_column}' not found in dataset. Please check column names.")

    # Convert target to numeric if it's not already
    data[target_column] = pd.Categorical(data[target_column]).codes

    # Split features and target
    X = data.drop([target_column], axis=1)
    y = data[target_column]
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train the model with hyperparameter tuning"""
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100,200] ,#[100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'scale_pos_weight': [1, 3, 5]  # Help with imbalanced classes
    }

    # Initialize the model
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=2,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    # Fit the model
    print("Training model with hyperparameter tuning...")
    grid_search.fit(X_train, y_train)

    # Print best parameters and score
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    # Return the best model
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print detailed metrics"""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Get feature importance
    feature_importance = model.feature_importances_
    print("\nFeature Importance:")
    for i, importance in enumerate(feature_importance):
        print(f"Feature {i}: {importance:.4f}")
    
    return accuracy

def save_model(model, accuracy, model_dir='models'):
    """Save the trained model with timestamp and accuracy"""
    # Create models directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Create filename with timestamp and accuracy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    accuracy_str = f"{accuracy:.4f}".replace(".", "")
    filename = f"model_{timestamp}_acc_{accuracy_str}.joblib"
    filepath = os.path.join(model_dir, filename)
    
    # Save the model
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")
    return filepath

def load_model(filepath):
    """Load a saved model"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model 
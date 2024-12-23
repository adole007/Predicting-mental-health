import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_histograms(data):
    """
    Plot histograms for all numeric columns in the dataset
    """
    # Create a copy and drop the Name column if it exists
    df = data.copy()
    if 'Name' in df.columns:
        df = df.drop('Name', axis=1)
    
    # Convert categorical columns to numeric
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
        'Chronic Medical Conditions',
        'History of Mental Illness'
    ]
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes

    # Create subplots for each column
    n_cols = 3
    n_rows = (len(df.columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()

    # Plot histogram for each column
    for idx, col in enumerate(df.columns):
        sns.histplot(data=df, x=col, ax=axes[idx])
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].tick_params(axis='x', rotation=45)

    # Remove empty subplots
    for idx in range(len(df.columns), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig('visualizations/feature_distributions.png')
    plt.close()

def plot_correlation_matrix(data):
    """
    Plot correlation matrix for all numeric columns
    """
    # Create a copy and drop the Name column if it exists
    df = data.copy()
    if 'Name' in df.columns:
        df = df.drop('Name', axis=1)
    
    # Convert categorical columns to numeric
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
        'Chronic Medical Conditions',
        'History of Mental Illness'
    ]
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes

    # Calculate correlation matrix
    corr_matrix = df.corr()

    # Create correlation matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f',
                square=True)
    plt.title('Feature Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('visualizations/correlation_matrix.png')
    plt.close()

def create_visualizations(data):
    """
    Create all visualizations
    """
    # Create visualizations directory if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
        
    # Create visualizations
    plot_histograms(data)
    plot_correlation_matrix(data)
    print("Visualizations have been saved to the 'visualizations' directory.") 
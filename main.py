from src.data_processing import load_data, preprocess_data
from src.visualization import create_visualizations
from src.model import prepare_data, train_model, evaluate_model, save_model

def main():
    # Load and process data
    data_path = "depression_data.csv"  # Specify the path to your data file
    processed_data = load_data(data_path)

     # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(processed_data)
    
    # Print data info for debugging
    print("\nDataset Info:")
    print(processed_data.info())
    print("\nColumn names:")
    print(processed_data.columns.tolist())
    
    # Prepare and train model
    X_train, X_test, y_train, y_test = prepare_data(processed_data)
    
    # Train and evaluate model
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Save the trained model
    model_path = save_model(model, accuracy)

if __name__ == "__main__":
    main()

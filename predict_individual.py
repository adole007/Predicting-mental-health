import pandas as pd
from src.model import load_model
from src.data_processing import preprocess_data
import os

def get_latest_model(model_dir='models'):
    """Get the most recently saved model"""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    if not model_files:
        raise FileNotFoundError("No model files found in models directory")
    
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    return os.path.join(model_dir, latest_model)

def predict_for_individual(name, data_path='depression_data.csv'):
    try:
        # Load the model using get_latest_model()
        model_path = get_latest_model()
        model = load_model(model_path)
        print(f"Model loaded from: {model_path}")

        # Load the full dataset
        df = pd.read_csv(data_path)
        
        # Find all people with matching name
        people_data = df[df['Name'].str.lower() == name.lower()]
        
        if len(people_data) == 0:
            return {'error': f"No data found for {name}"}
        
        results = []
        # Process each person's data
        for idx, person_data in people_data.iterrows():
            # Convert single row to DataFrame
            person_df = pd.DataFrame([person_data])
            
            # Preprocess the data
            processed_data = preprocess_data(person_df, for_training=False)
            
            # Make prediction
            prediction = model.predict(processed_data)
            
            # Store result
            results.append({
                'name': person_data['Name'],
                'prediction': "At risk" if prediction[0] == 1 else "Not at risk",
                'confidence': None,
                'details': person_data.to_dict()
            })
        
        return {
            'multiple': len(results) > 1,
            'results': results
        }
        
    except Exception as e:
        print(f"\nError: Error during prediction: {str(e)}")
        return {'error': str(e)}

def main():
    """Interactive command-line interface for making predictions"""
    while True:
        print("\n=== Mental Health Risk Prediction ===")
        name = input("\nEnter name to search (or 'quit' to exit): ")
        
        if name.lower() == 'quit':
            break
        
        result = predict_for_individual(name)
        
        if 'error' in result:
            print(f"\nError: {result['error']}")
        else:
            if result['multiple']:
                print(f"\nFound {len(result['results'])} entries for {name}:")
                
            for idx, person_result in enumerate(result['results'], 1):
                if result['multiple']:
                    print(f"\n--- Entry {idx} ---")
                print("\nPrediction Results:")
                print(f"Name: {person_result['name']}")
                print(f"Prediction: {person_result['prediction']}")
                if person_result['confidence']:
                    print(f"Confidence: {person_result['confidence']}")
                print("\nIndividual Details:")
                for key, value in person_result['details'].items():
                    if key != 'Name':  # Skip printing name again
                        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
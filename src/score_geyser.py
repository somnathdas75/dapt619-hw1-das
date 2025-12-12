import pandas as pd
import joblib
import os

def score_geyser():
    # Load the model
    model_path = './models/linear_regression_pipeline.joblib'
    pipeline = joblib.load(model_path)
    
    # Load the data - it's a TSV file
    data_path = './data/raw/geyser.tsv'
    df = pd.read_csv(data_path, sep='\t')
    
    # Make predictions
    predictions = pipeline.predict(df[['eruptions']])
    df['predicted_waiting'] = predictions
    
    # Create output directory if it doesn't exist
    output_dir = './data/scored'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save scored data
    output_path = os.path.join(output_dir, 'geyser_scored.csv')
    df[['eruptions', 'waiting', 'predicted_waiting']].to_csv(output_path, index=False)
    
    print(f"Scored dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    score_geyser()

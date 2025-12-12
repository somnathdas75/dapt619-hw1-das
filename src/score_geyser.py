import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

def score_geyser():
    # Load the model
    model_path = './models/linear_regression_pipeline.joblib'
    pipeline = joblib.load(model_path)
    
    # Load the data
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
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['waiting'], df['predicted_waiting'], alpha=0.6)
    plt.plot([df['waiting'].min(), df['waiting'].max()], 
             [df['waiting'].min(), df['waiting'].max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Waiting Time at Yellowstone')
    plt.ylabel('Predicted Waiting Time for Old Faithfull Geyser')
    plt.title('Somnath das Geyser Predictions: Actual vs Predicted Waiting Times')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_dir = './plots'
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'geyser_predictions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scored dataset saved to: {output_path}")
    print(f"Plot saved to: {plot_path}")
    print(f"Shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    score_geyser()

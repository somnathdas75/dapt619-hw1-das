import pandas as pd
import joblib
import numpy as np
import os

def test_scoring():
    # Load the model
    model_path = './models/linear_regression_pipeline.joblib'
    pipeline = joblib.load(model_path)
    
    # Test data
    df = pd.DataFrame({"eruptions": [1.5, 2.0, 3.0]})
    
    # Make predictions
    preds = pipeline.predict(df[["eruptions"]])
    
    # Test 1: Number of predictions matches number of inputs
    assert len(preds) == len(df), \
        f"Expected {len(df)} predictions, got {len(preds)}"
    
    # Test 2: All predictions are finite (not NaN or inf)
    assert np.all(np.isfinite(preds)), \
        "Some predictions are not finite (NaN or inf)"
    
    # Test 3: All predictions are positive
    assert np.all(preds > 0), \
        "Some predictions are not positive (greater than zero)"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_scoring()

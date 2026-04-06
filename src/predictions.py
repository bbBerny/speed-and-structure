# =============================================================================
# PREDICTION NOTEBOOK - Updated for Contest Submission Format
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.loaders import load_data_array_test

# Contest submission function
def save_submission_batch(predictions_dict: dict, submission_path: str):
    """Function to create submission file out of a batch of predictions"""
    try:
        if os.path.exists(submission_path):
            submission = dict(np.load(submission_path))
        else:
            submission = dict({})
    except Exception as e:
        print(f"Warning: could not load existing submission: {e}. Starting fresh.")
        submission = dict({})
    
    submission.update(predictions_dict)
    np.savez(submission_path, **submission)
    return

def load_trained_model(model_path):
    """Load the trained model"""
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
    
    # Print model input shapes for debugging
    print("Model input shapes:")
    for i, input_layer in enumerate(model.inputs):
        print(f"  Input {i} ({input_layer.name}): {input_layer.shape}")
    
    return model

def load_test_data():
    """Load test data using Option 1 (NumPy arrays)"""
    print("Loading test data...")
    
    # Load test features
    test_features_path = os.path.join(repo_root, 'data', 'processed', 'test_features.csv')
    test_features = pd.read_csv(test_features_path)
    test_ids = test_features['sample_id'].values
    
    print("Test features columns:", test_features.columns.tolist())
    print("Test features shape:", test_features.shape)
    
    # Extract wide features (remove sample_id)
    X_wide_test = test_features.drop(columns=['sample_id']).values.astype(np.float32)
    
    print("Wide features shape after removing sample_id:", X_wide_test.shape)
    print("Expected by model: (batch_size, 6)")
    
    # Check if we need to select specific columns to match training
    if X_wide_test.shape[1] != 6:
        print(f"❌ Mismatch! Test has {X_wide_test.shape[1]} features, model expects 6")
        print("Available columns (excluding sample_id):")
        feature_cols = [col for col in test_features.columns if col != 'sample_id']
        for i, col in enumerate(feature_cols):
            print(f"  {i}: {col}")
        
        # You might need to select the first 6 columns or specific ones
        print("Using first 6 features for now...")
        X_wide_test = X_wide_test[:, :6]
        print("Adjusted wide features shape:", X_wide_test.shape)
    
    # Load all deep features
    print(f"Loading {len(test_ids)} test samples...")
    X_deep_list = []
    
    for i, sample_id in enumerate(test_ids):
        if i % 50 == 0:
            print(f"  Loaded {i}/{len(test_ids)} samples")
            
        try:
            # Load deep features for this sample
            X_deep_single = load_data_array_test([sample_id])  # Ignore targets
            X_deep_list.append(X_deep_single[0])  # Remove batch dimension
        except Exception as e:
            print(f"  Error loading sample {sample_id}: {e}")
            continue
    
    # Stack into arrays
    X_deep_test = np.stack(X_deep_list).astype(np.float32)
    
    print(f"✅ Test data loaded!")
    print(f"X_wide_test shape: {X_wide_test.shape}")
    print(f"X_deep_test shape: {X_deep_test.shape}")
    
    return X_wide_test, X_deep_test, test_ids

def make_predictions_and_save(model, X_wide_test, X_deep_test, test_ids, 
                             submission_path, batch_size=8):
    """Make predictions and save using contest format"""
    print("Making predictions and saving to contest format...")
    
    # Create submission directory if it doesn't exist
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    
    # Process in batches to manage memory
    num_samples = len(test_ids)
    
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch_indices = range(i, end_idx)
        
        # Get batch data
        X_wide_batch = X_wide_test[batch_indices]
        X_deep_batch = X_deep_test[batch_indices]
        batch_ids = test_ids[batch_indices]
        
        # Make predictions for this batch
        batch_predictions = model.predict(
            [X_wide_batch, X_deep_batch], 
            batch_size=len(batch_indices), 
            verbose=0
        )
        
        # Create a dictionary for this batch's predictions
        batch_predictions_dict = {}
        
        # Save each prediction using contest format
        for j, sample_id in enumerate(batch_ids):
            prediction = batch_predictions[j]  # Shape: (300, 1259)
            
            # Ensure prediction is 2D as expected
            assert prediction.shape == (300, 1259), f"Expected (300, 1259), got {prediction.shape}"
            
            # Convert to float64 as required by contest
            batch_predictions_dict[sample_id] = prediction.astype(np.float64)
            
        # Save batch to contest submission file all at once
        save_submission_batch(batch_predictions_dict, submission_path)
        
        print(f"Processed batch {i//batch_size + 1}/{(num_samples-1)//batch_size + 1}")
    
    print(f"✅ All predictions saved to {submission_path}")

def verify_submission(submission_path, expected_count):
    """Verify the submission file contains all expected samples"""
    print(f"Verifying submission file...")
    
    try:
        submission = dict(np.load(submission_path))
        print(f"Submission contains {len(submission)} samples")
        print(f"Expected: {expected_count}")
        
        # Check a few sample shapes
        sample_keys = list(submission.keys())[:3]
        for key in sample_keys:
            shape = submission[key].shape
            print(f"Sample {key} shape: {shape}")
            
        if len(submission) == expected_count:
            print("✅ Submission verification passed!")
        else:
            print("❌ Submission count mismatch!")
            
    except Exception as e:
        print(f"❌ Error verifying submission: {e}")

def main():
    """Main prediction pipeline"""
    print("=" * 60)
    print("MODEL 02 MLP - CONTEST SUBMISSION PIPELINE")
    print("=" * 60)
    
    # 1. Load trained model
    model_path = os.path.join(repo_root, 'models', 'model_02_mlp.keras')
    model = load_trained_model(model_path)
    
    # 2. Load test data
    X_wide_test, X_deep_test, test_ids = load_test_data()
    
    # 3. Make predictions and save in contest format
    submission_path = os.path.join(repo_root, 'submissions', 'model_02_mlp_submission.npz')
    make_predictions_and_save(
        model, X_wide_test, X_deep_test, test_ids, 
        submission_path, batch_size=8
    )
    
    # 4. Verify submission
    verify_submission(submission_path, len(test_ids))
    
    print("\n" + "=" * 60)
    print("CONTEST SUBMISSION PIPELINE COMPLETE")
    print("=" * 60)
    
    return submission_path

if __name__ == "__main__":
    submission_path = main()
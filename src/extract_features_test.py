#!/usr/bin/env python3
"""
Test Data Preprocessing Pipeline
================================

Applies the same preprocessing pipeline to test data using saved scaler and reducers.

Usage:
    python test_preprocessing.py

Requirements:
    - Must run train preprocessing first to save scaler/reducers
    - Test data structure should match training data structure

Output:
    - ../data/processed/test_features.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.decomposition import PCA

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.loaders import load_gather
from src.features import (
    compute_time_stats, 
    compute_envelope_stats, 
    compute_spectral_stats, 
    compute_spatial_stats,
    pick_first_breaks
)

# Configuration
TEST_DATA_DIR = '../data/raw/test_data'  # Adjust path as needed
OUTPUT_PATH = '../data/processed/test_features.csv'
SCALER_PATH = '../data/processed/scaler.pkl'
PCA_PATH = '../data/processed/pca_reducer.pkl'
UMAP_PATH = '../data/processed/umap_reducer.pkl'

def get_test_sample_ids():
    """Get list of test sample IDs"""
    if not os.path.exists(TEST_DATA_DIR):
        raise FileNotFoundError(f"Test data directory not found: {TEST_DATA_DIR}")
    
    test_ids = [d for d in os.listdir(TEST_DATA_DIR) 
                if os.path.isdir(os.path.join(TEST_DATA_DIR, d))]
    
    print(f"Found {len(test_ids)} test samples")
    return test_ids

def extract_test_features(sample_ids, data_dir):
    """Extract features from test data (same as training pipeline)"""
    print(f"Extracting features from {len(sample_ids)} test samples...")
    
    records = []
    for i, sid in enumerate(sample_ids):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(sample_ids)}: {sid}")
            
        try:
            path = os.path.join(data_dir, sid)
            
            # Load gathers: shape (5 shots, time, receivers)
            gathers = np.stack([
                np.load(os.path.join(path, f'receiver_data_src_{src}.npy'))
                for src in (1, 75, 150, 225, 300)
            ], axis=0)

            # Time-domain statistics
            mean_amp, std_amp, rms_amp = compute_time_stats(gathers)
            
            # Envelope statistics
            env_mean, env_max = compute_envelope_stats(gathers)
            
            # Spectral statistics
            dom_freq, spec_cent = compute_spectral_stats(gathers)
            
            # First breaks
            first_break = pick_first_breaks(gathers)

            records.append({
                'sample_id': sid,
                'mean_amp': mean_amp,
                'std_amp': std_amp,
                'rms_amp': rms_amp,
                'env_mean': env_mean,
                'spec_cent': spec_cent,
                'first_break': first_break
            })
            
        except Exception as e:
            print(f"  Error processing {sid}: {e}")
            continue
    
    return pd.DataFrame(records)

def apply_saved_preprocessing(df):
    """Apply saved scaler and dimensionality reducers"""
    print("Applying saved preprocessing...")
    
    # Load saved preprocessing objects
    try:
        scaler = joblib.load(SCALER_PATH)
        pca = joblib.load(PCA_PATH)
        umap_reducer = joblib.load(UMAP_PATH)
        print("  Loaded saved scaler and reducers")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Preprocessing objects not found. Run training preprocessing first. {e}")
    
    # Features to normalize (must match training exactly)
    feature_cols = ['mean_amp', 'std_amp', 'rms_amp', 'env_mean', 'spec_cent', 'first_break']
    
    # Check for missing features
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"  Warning: Missing features {missing_cols}, filling with 0")
        for col in missing_cols:
            df[col] = 0.0
    
    # Handle NaN values
    df_clean = df.copy()
    print("  Checking for NaN values...")
    for col in feature_cols:
        nan_count = df_clean[col].isna().sum()
        if nan_count > 0:
            print(f"    Found {nan_count} NaN values in {col}, filling with 0")
            df_clean[col] = df_clean[col].fillna(0.0)
    
    # Apply normalization (transform only, no fitting)
    df_processed = df_clean.copy()
    df_processed[feature_cols] = scaler.transform(df_clean[feature_cols])
    
    # Apply PCA
    pca_coords = pca.transform(df_processed[feature_cols])
    df_processed['pca1'] = pca_coords[:, 0]
    df_processed['pca2'] = pca_coords[:, 1]
    
    # Apply UMAP
    umap_coords = umap_reducer.transform(df_processed[feature_cols])
    df_processed['umap1'] = umap_coords[:, 0]
    df_processed['umap2'] = umap_coords[:, 1]
    
    print("  Applied normalization and dimensionality reduction")
    return df_processed

def create_final_test_features(df):
    """Select final features matching training set"""
    final_features = [
        'sample_id',
        'mean_amp', 'std_amp', 'rms_amp',
        'env_mean', 'spec_cent',
        'first_break',
        'umap1', 'umap2'
    ]
    
    available_features = [f for f in final_features if f in df.columns]
    return df[available_features].copy()

def main():
    """Main test preprocessing pipeline"""
    print("=" * 60)
    print("TEST DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # 1. Get test sample IDs
    print("\n1. Loading test sample IDs...")
    test_ids = get_test_sample_ids()
    
    # 2. Extract features
    print("\n2. Extracting features...")
    df = extract_test_features(test_ids, TEST_DATA_DIR)
    
    # 3. Apply saved preprocessing
    print("\n3. Applying saved preprocessing...")
    df_processed = apply_saved_preprocessing(df)
    
    # 4. Create final feature set
    print("\n4. Creating final feature set...")
    final_df = create_final_test_features(df_processed)
    
    # 5. Save results
    print("\n5. Saving results...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"   Saved to: {OUTPUT_PATH}")
    print(f"   Final shape: {final_df.shape}")
    
    print("\n" + "=" * 60)
    print("TEST PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Test samples processed: {len(final_df)}")
    print(f"Features: {final_df.shape[1] - 1}")  # Exclude sample_id
    
    return final_df

if __name__ == "__main__":
    df = main()
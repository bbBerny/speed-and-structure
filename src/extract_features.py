#!/usr/bin/env python3
"""
Seismic Data Preprocessing Pipeline
===================================

Extracts features from raw seismic data and creates final feature matrix for training.
Based on notebooks 02_feature_engineering.ipynb and 03_clustering_pca.ipynb

Usage:
    python preprocessing_pipeline.py

Output:
    - ../data/processed/final_features.csv (features ready for model training)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.decomposition import PCA
import joblib

# Add current directory and parent to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

try:
    # Try importing from src (if running from root)
    from src.loaders import shuffle_and_split_data, get_training_ids, load_gather
    from src.features import (
        compute_time_stats, 
        compute_envelope_stats, 
        compute_spectral_stats, 
        compute_spatial_stats,
        compute_metrics,
        pick_first_breaks
    )
except ImportError:
    # Try importing directly (if running from src/)
    from loaders import shuffle_and_split_data, get_training_ids, load_gather
    from features import (
        compute_time_stats, 
        compute_envelope_stats, 
        compute_spectral_stats, 
        compute_spatial_stats,
        compute_metrics,
        pick_first_breaks
    )

# Configuration
DATA_DIR = os.path.join(parent_dir, 'data', 'raw', 'train_data')
OUTPUT_PATH = os.path.join(parent_dir, 'data', 'processed', 'final_features.csv')

def extract_basic_features(sample_ids):
    """Extract basic features from seismic data"""
    print(f"Extracting basic features from {len(sample_ids)} samples...")
    
    records = []
    for i, sid in enumerate(sample_ids):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(sample_ids)}: {sid}")
            
        try:
            path = os.path.join(DATA_DIR, sid)
            
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

            records.append({
                'sample_id': sid,
                'mean_amp': mean_amp,
                'std_amp': std_amp,
                'rms_amp': rms_amp,
                'env_mean': env_mean,
                'env_max': env_max,  # Will be dropped later
                'dom_freq': dom_freq,  # Will be dropped later
                'spec_cent': spec_cent
            })
            
        except Exception as e:
            print(f"  Error processing {sid}: {e}")
            continue
    
    return pd.DataFrame(records)

def extract_advanced_features(df):
    """Extract first breaks and other advanced features"""
    print("Extracting advanced features (first breaks)...")
    
    first_breaks = []
    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"  Processing {i}/{len(df)}: {row['sample_id']}")
            
        try:
            sid = row['sample_id']
            gather = load_gather(sid)
            fb = pick_first_breaks(gather)
            first_breaks.append(fb)
        except Exception as e:
            print(f"  Error extracting first break for {sid}: {e}")
            first_breaks.append(np.nan)
    
    df['first_break'] = first_breaks
    return df

def remove_redundant_features(df):
    """Remove highly correlated features"""
    print("Removing redundant features...")
    
    # Remove features identified as not important
    features_to_drop = ['env_max', 'dom_freq']
    existing_to_drop = [f for f in features_to_drop if f in df.columns]
    if existing_to_drop:
        df = df.drop(columns=existing_to_drop)
        print(f"  Dropped {existing_to_drop} (not important)")
    
    return df

def normalize_features(df, train_mask):
    """Normalize features using StandardScaler fitted on training data only"""
    print("Normalizing features...")
    
    # Features to normalize (exclude sample_id and split)
    feature_cols = [col for col in df.columns 
                   if col not in ['sample_id', 'split']]
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    train_data = df.loc[train_mask, feature_cols]
    scaler.fit(train_data)
    
    # Apply to all data
    df_normalized = df.copy()
    df_normalized[feature_cols] = scaler.transform(df[feature_cols])  # Use transform, not fit_transform
    
    print(f"  Normalized {len(feature_cols)} features")
    
    # Save the scaler for test data preprocessing
    scaler_path = os.path.join(parent_dir, 'data', 'processed', 'scaler.pkl')
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler to {scaler_path}")
    
    return df_normalized

def compute_embeddings(df):
    """Compute PCA and UMAP embeddings"""
    print("Computing dimensionality reduction embeddings...")
    
    # Get feature matrix (exclude sample_id and split)
    feature_cols = [col for col in df.columns 
                   if col not in ['sample_id', 'split']]
    X = df[feature_cols].values
    
    # PCA (2 components)
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X)
    df['pca1'] = pca_coords[:, 0]
    df['pca2'] = pca_coords[:, 1]
    
    # UMAP (2 components)
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    umap_coords = umap_reducer.fit_transform(X)
    df['umap1'] = umap_coords[:, 0]
    df['umap2'] = umap_coords[:, 1]
    
    # Save PCA and UMAP for test data preprocessing
    pca_path = os.path.join(parent_dir, 'data', 'processed', 'pca_reducer.pkl')
    umap_path = os.path.join(parent_dir, 'data', 'processed', 'umap_reducer.pkl')
    joblib.dump(pca, pca_path)
    joblib.dump(umap_reducer, umap_path)
    print(f"  Saved PCA reducer to {pca_path}")
    print(f"  Saved UMAP reducer to {umap_path}")
    
    print("  Added PCA and UMAP embeddings")
    return df

def create_final_features(df):
    """Select final features for model training"""
    print("Creating final feature set...")
    
    # Keep essential features based on your analysis
    final_features = [
        'sample_id', 'split',
        'mean_amp', 'std_amp', 'rms_amp',
        'env_mean', 'spec_cent',
        'first_break',
        'umap1', 'umap2'
    ]
    
    # Only keep features that exist in the dataframe
    available_features = [f for f in final_features if f in df.columns]
    missing_features = [f for f in final_features if f not in df.columns]
    
    if missing_features:
        print(f"  Warning: Missing features {missing_features}")
    
    final_df = df[available_features].copy()
    print(f"  Final feature set: {len(available_features)} features")
    
    return final_df

def main():
    """Main preprocessing pipeline"""
    print("=" * 60)
    print("SEISMIC DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # 1. Get sample IDs and create train/val split
    print("\n1. Loading sample IDs and creating splits...")
    _, all_ids = get_training_ids()
    train_ids, val_ids = shuffle_and_split_data(all_ids, val_ratio=0.2)
    
    print(f"   Total samples: {len(all_ids)}")
    print(f"   Train samples: {len(train_ids)}")
    print(f"   Val samples: {len(val_ids)}")
    
    # 2. Extract basic features
    print("\n2. Extracting basic features...")
    all_sample_ids = train_ids + val_ids
    df = extract_basic_features(all_sample_ids)
    
    # Add split information
    df['split'] = df['sample_id'].apply(
        lambda x: 'train' if x in train_ids else 'val'
    )
    
    # 3. Extract advanced features
    print("\n3. Extracting advanced features...")
    df = extract_advanced_features(df)
    
    # 4. Remove redundant features
    print("\n4. Feature selection...")
    df = remove_redundant_features(df)
    
    # 5. Normalize features
    print("\n5. Feature normalization...")
    train_mask = df['split'] == 'train'
    df = normalize_features(df, train_mask)
    
    # 6. Compute embeddings
    print("\n6. Computing embeddings...")
    df = compute_embeddings(df)
    
    # 7. Create final feature set
    print("\n7. Creating final feature set...")
    final_df = create_final_features(df)
    
    # 8. Save results
    print("\n8. Saving results...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"   Saved to: {OUTPUT_PATH}")
    print(f"   Final shape: {final_df.shape}")
    print(f"   Features: {list(final_df.columns)}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Train samples: {len(final_df[final_df['split'] == 'train'])}")
    print(f"Val samples: {len(final_df[final_df['split'] == 'val'])}")
    print(f"Features: {final_df.shape[1] - 2}")  # Exclude sample_id and split
    
    return final_df

if __name__ == "__main__":
    df = main()
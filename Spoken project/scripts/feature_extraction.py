import numpy as np
import librosa

def extract_features(audio_path, sr=16000, n_mfcc=13):
    """
    Extracts 39-dimensional MFCC features (static, delta, delta-delta) from an audio file.
    
    Args:
        audio_path (str): Path to the audio file.
        sr (int): Sampling rate. Defaults to 16,000 Hz.
        n_mfcc (int): Number of MFCCs to extract. Defaults to 13.
    
    Returns:
        ndarray: 39-dimensional feature vector (mean of static, delta, and delta-delta MFCCs).
    """
    print(f"Loading audio file: {audio_path} with sampling rate: {sr}")
    # Load preprocessed audio
    audio, sr = librosa.load(audio_path, sr=sr)
    print(f"Audio loaded. Length: {len(audio)} samples. Sampling Rate: {sr}")
    
    # Compute static MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    print(f"Computed MFCCs. Shape: {mfcc.shape}")
    
    # Compute delta (first derivative) MFCCs
    delta_mfcc = librosa.feature.delta(mfcc)
    print(f"Computed Delta MFCCs. Shape: {delta_mfcc.shape}")
    
    # Compute delta-delta (second derivative) MFCCs
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    print(f"Computed Delta-Delta MFCCs. Shape: {delta2_mfcc.shape}")
    
    # Stack static, delta, and delta-delta MFCCs
    combined_mfcc = np.vstack((mfcc, delta_mfcc, delta2_mfcc))
    print(f"Stacked MFCCs (Static, Delta, Delta-Delta). Shape: {combined_mfcc.shape}")
    
    # Compute the mean of each coefficient across all frames
    mfcc_mean = np.mean(combined_mfcc, axis=1)
    print(f"Computed Mean MFCCs. Shape: {mfcc_mean.shape}")
    
     # Compute the variance of each coefficient across all frames
    mfcc_variance = np.var(combined_mfcc, axis=1)
    print(f" - Computed Variance of MFCCs. Shape: {mfcc_variance.shape}")  # (3*n_mfcc,)
    
     # Concatenate mean and variance to form a 78-dimensional feature vector
    mfcc_features = np.concatenate((mfcc_mean, mfcc_variance))
    print(f" - Concatenated Mean and Variance. Feature Vector Shape: {mfcc_features.shape}")  # (6*n_mfcc,)

    return mfcc_features  # 78-dimensional vector

if __name__ == "__main__":
    import pandas as pd
    import os
    
    # Load training metadata
    print("Loading testing metadata...")
    train_df = pd.read_csv('test_metadata.csv')
    print(f"Training metadata loaded. Number of rows: {len(train_df)}")
    
    # Initialize lists to store features and labels
    X_test, y_test = [], []
    
    for idx, row in train_df.iterrows():
        # Construct the full path to the audio file
        audio_path = os.path.join(r'../data', 'processed', 'test', row['filename'])
        print(f"Processing file {idx+1}/{len(train_df)}: {audio_path}")
        
        # Extract features and append to the list
        feats = extract_features(audio_path)
        X_test.append(feats)
        y_test.append(row['ethnicity'])
        print(f"Extracted features for file {row['filename']}.")
    
    # Convert lists to numpy arrays for efficient storage and processing
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print(f"Feature array shape: {X_test.shape}")
    print(f"Labels array shape: {y_test.shape}")
    
    # Save features for fast loading later
    np.save(r'../data/Testing_features/X_test.npy', X_test)
    print("Saved X_test features to 'data/Testing_features/X_test.npy'.")
    np.save(r'../data/Testing_features/y_test.npy', y_test)
    print("Saved y_test labels to 'data/Testing_features/y_test.npy'.")

"""
Data manifest utilities for label integrity and single source of truth.

Provides functions to load, validate, and encode labels from manifest CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Dict, Tuple, List, Optional


def load_manifest(manifest_path: str) -> Tuple[Dict[str, str], List[str]]:
    """
    Load manifest CSV file with filename and label columns.
    
    Parameters:
    -----------
    manifest_path : str
        Path to manifest CSV file with columns: filename, label
        
    Returns:
    --------
    tuple : (manifest_dict, class_list)
        - manifest_dict: Dictionary mapping filename to label
        - class_list: Ordered list of unique class names (sorted)
        
    Raises:
    -------
    FileNotFoundError: If manifest file doesn't exist
    ValueError: If manifest file doesn't have required columns
    """
    if not Path(manifest_path).exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    df = pd.read_csv(manifest_path)
    
    # Check required columns
    if 'filename' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"Manifest must have 'filename' and 'label' columns. Found: {df.columns.tolist()}")
    
    # Create dictionary mapping filename to label
    manifest_dict = dict(zip(df['filename'], df['label']))
    
    # Get ordered class list (sorted unique labels)
    class_list = sorted(df['label'].unique().tolist())
    
    return manifest_dict, class_list


def encode_labels(labels: np.ndarray, encoder_path: Optional[str] = None) -> Tuple[np.ndarray, LabelEncoder, List[str]]:
    """
    Encode string labels to integers (0..K-1) using LabelEncoder.
    
    Parameters:
    -----------
    labels : np.ndarray
        Array of string labels
    encoder_path : str, optional
        Path to save encoder.pkl. If None, encoder is not saved.
        
    Returns:
    --------
    tuple : (y_encoded, encoder, class_names)
        - y_encoded: Encoded labels as integers (0..K-1)
        - encoder: Fitted LabelEncoder
        - class_names: List of class names in encoder order
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(labels)
    class_names = encoder.classes_.tolist()
    
    # Save encoder if path provided
    if encoder_path:
        Path(encoder_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoder, encoder_path)
    
    return y_encoded, encoder, class_names


def validate_labels(df: pd.DataFrame, manifest_dict: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """
    Validate that all filenames in dataframe exist in manifest and labels match.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'filename' column (and optionally 'label' column)
    manifest_dict : dict
        Dictionary mapping filename to label from manifest
        
    Returns:
    --------
    list : List of mismatches as tuples (filename, df_label, manifest_label)
        Empty list if all labels match
        
    Raises:
    -------
    ValueError: If any filename is missing from manifest
    """
    mismatches = []
    missing_files = []
    
    # Check if filename column exists
    if 'filename' not in df.columns:
        raise ValueError("DataFrame must have 'filename' column")
    
    # Check all filenames exist in manifest
    for filename in df['filename']:
        if filename not in manifest_dict:
            missing_files.append(filename)
    
    if missing_files:
        raise ValueError(f"Files missing from manifest: {missing_files[:10]}{'...' if len(missing_files) > 10 else ''}")
    
    # Check label matches if label column exists in df
    if 'label' in df.columns:
        for idx, row in df.iterrows():
            filename = row['filename']
            df_label = str(row['label'])
            manifest_label = str(manifest_dict[filename])
            
            if df_label != manifest_label:
                mismatches.append((filename, df_label, manifest_label))
    
    return mismatches


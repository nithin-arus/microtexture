import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import re
from typing import Tuple, Dict, List, Optional
import warnings


class SampleAwareSplitter:
    """
    Sample-aware data splitting to prevent data leakage in texture analysis.
    
    Ensures that all patches/windows from the same fabric sample are assigned
    to the same split (train/validation/test), preventing information leakage
    that could lead to overly optimistic performance estimates.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the sample-aware splitter.
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducible splitting
        """
        self.random_state = random_state
        self.sample_mapping = {}
        
    def extract_sample_id(self, filename: str, path: str = None) -> str:
        """
        Extract sample identifier from filename or path.
        
        Assumes filenames follow patterns like:
        - material_date_time.jpg
        - material_sample01_patch001.jpg
        - /path/to/cotton/cotton_001.jpg
        
        Parameters:
        -----------
        filename : str
            Name of the image file
        path : str, optional
            Full path to the image file
            
        Returns:
        --------
        str : Sample identifier
        """
        # Use full path if available for better sample identification
        full_name = path if path else filename
        
        # Extract directory name as potential sample indicator
        if path:
            parent_dir = Path(path).parent.name
            if parent_dir not in ['images', 'capture']:
                # Directory name likely indicates sample type
                sample_base = parent_dir
            else:
                sample_base = Path(path).stem
        else:
            sample_base = Path(filename).stem
            
        # Remove common suffixes that indicate patches/windows
        sample_base = re.sub(r'_patch\d+|_window\d+|_\d{14}|_\d{8}_\d{6}', '', sample_base)
        
        # Extract base sample name (everything before final timestamp or patch number)
        sample_parts = sample_base.split('_')
        if len(sample_parts) >= 2:
            # Keep material type and sample identifier, remove timestamps
            filtered_parts = []
            for part in sample_parts:
                # Skip pure numeric parts that look like timestamps
                if not (part.isdigit() and len(part) >= 6):
                    filtered_parts.append(part)
            sample_id = '_'.join(filtered_parts[:2]) if len(filtered_parts) >= 2 else '_'.join(filtered_parts)
        else:
            sample_id = sample_base
            
        return sample_id
    
    def create_sample_mapping(self, features_df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Create mapping from sample IDs to row indices.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            DataFrame containing features with filename/path columns
            
        Returns:
        --------
        Dict[str, List[int]] : Mapping from sample_id to list of row indices
        """
        sample_mapping = {}
        
        for idx, row in features_df.iterrows():
            filename = row.get('filename', '')
            path = row.get('path', '')
            
            sample_id = self.extract_sample_id(filename, path)
            
            if sample_id not in sample_mapping:
                sample_mapping[sample_id] = []
            sample_mapping[sample_id].append(idx)
            
        self.sample_mapping = sample_mapping
        
        # Print sample statistics
        sample_sizes = [len(indices) for indices in sample_mapping.values()]
        print(f"Identified {len(sample_mapping)} unique samples")
        print(f"Sample sizes - Min: {min(sample_sizes)}, Max: {max(sample_sizes)}, "
              f"Mean: {np.mean(sample_sizes):.1f}")
        
        return sample_mapping
    
    def split_by_samples(self, 
                        features_df: pd.DataFrame, 
                        test_size: float = 0.2, 
                        val_size: float = 0.1,
                        stratify_column: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data by samples, ensuring no sample appears in multiple splits.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            DataFrame containing features
        test_size : float
            Proportion of samples for testing
        val_size : float
            Proportion of samples for validation
        stratify_column : str, optional
            Column name for stratified splitting
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray] : Train, validation, test indices
        """
        # Create sample mapping
        sample_mapping = self.create_sample_mapping(features_df)
        
        # Get list of unique samples
        sample_ids = list(sample_mapping.keys())
        
        # Prepare stratification if requested
        if stratify_column and stratify_column in features_df.columns:
            # Create sample-level labels for stratification
            sample_labels = []
            for sample_id in sample_ids:
                # Use the most common label for this sample
                sample_indices = sample_mapping[sample_id]
                labels_for_sample = features_df.iloc[sample_indices][stratify_column]
                most_common_label = labels_for_sample.mode().iloc[0] if not labels_for_sample.empty else 0
                sample_labels.append(most_common_label)
            sample_labels = np.array(sample_labels)
        else:
            sample_labels = None
            
        # Split samples (not individual rows)
        if sample_labels is not None:
            try:
                train_samples, temp_samples, _, temp_labels = train_test_split(
                    sample_ids, sample_labels, 
                    test_size=(test_size + val_size), 
                    random_state=self.random_state,
                    stratify=sample_labels
                )
                
                # Further split temp into validation and test
                val_ratio = val_size / (test_size + val_size)
                val_samples, test_samples = train_test_split(
                    temp_samples, 
                    test_size=(1 - val_ratio),
                    random_state=self.random_state,
                    stratify=temp_labels
                )
            except ValueError as e:
                print(f"Warning: Stratified splitting failed ({e}), using random splitting")
                train_samples, temp_samples = train_test_split(
                    sample_ids, 
                    test_size=(test_size + val_size), 
                    random_state=self.random_state
                )
                val_ratio = val_size / (test_size + val_size)
                val_samples, test_samples = train_test_split(
                    temp_samples, 
                    test_size=(1 - val_ratio),
                    random_state=self.random_state
                )
        else:
            train_samples, temp_samples = train_test_split(
                sample_ids, 
                test_size=(test_size + val_size), 
                random_state=self.random_state
            )
            val_ratio = val_size / (test_size + val_size)
            val_samples, test_samples = train_test_split(
                temp_samples, 
                test_size=(1 - val_ratio),
                random_state=self.random_state
            )
            
        # Convert sample splits to row indices
        train_indices = []
        val_indices = []
        test_indices = []
        
        for sample_id in train_samples:
            train_indices.extend(sample_mapping[sample_id])
            
        for sample_id in val_samples:
            val_indices.extend(sample_mapping[sample_id])
            
        for sample_id in test_samples:
            test_indices.extend(sample_mapping[sample_id])
            
        # Verify no overlap
        train_set = set(train_indices)
        val_set = set(val_indices)
        test_set = set(test_indices)
        
        assert len(train_set & val_set) == 0, "Train and validation sets overlap!"
        assert len(train_set & test_set) == 0, "Train and test sets overlap!"
        assert len(val_set & test_set) == 0, "Validation and test sets overlap!"
        
        print(f"Sample-aware split completed:")
        print(f"  Train: {len(train_samples)} samples ({len(train_indices)} patches)")
        print(f"  Val:   {len(val_samples)} samples ({len(val_indices)} patches)")
        print(f"  Test:  {len(test_samples)} samples ({len(test_indices)} patches)")
        
        return np.array(train_indices), np.array(val_indices), np.array(test_indices)
    
    def validate_split(self, features_df: pd.DataFrame, 
                      train_idx: np.ndarray, 
                      val_idx: np.ndarray, 
                      test_idx: np.ndarray) -> bool:
        """
        Validate that the split has no sample leakage.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            DataFrame containing features
        train_idx, val_idx, test_idx : np.ndarray
            Indices for each split
            
        Returns:
        --------
        bool : True if split is valid (no leakage)
        """
        # Extract sample IDs for each split
        train_samples = set()
        val_samples = set()
        test_samples = set()
        
        for idx in train_idx:
            row = features_df.iloc[idx]
            sample_id = self.extract_sample_id(row.get('filename', ''), row.get('path', ''))
            train_samples.add(sample_id)
            
        for idx in val_idx:
            row = features_df.iloc[idx]
            sample_id = self.extract_sample_id(row.get('filename', ''), row.get('path', ''))
            val_samples.add(sample_id)
            
        for idx in test_idx:
            row = features_df.iloc[idx]
            sample_id = self.extract_sample_id(row.get('filename', ''), row.get('path', ''))
            test_samples.add(sample_id)
            
        # Check for overlaps
        train_val_overlap = train_samples & val_samples
        train_test_overlap = train_samples & test_samples
        val_test_overlap = val_samples & test_samples
        
        if train_val_overlap:
            print(f"ERROR: Train-Val sample overlap: {train_val_overlap}")
            
        if train_test_overlap:
            print(f"ERROR: Train-Test sample overlap: {train_test_overlap}")
            
        if val_test_overlap:
            print(f"ERROR: Val-Test sample overlap: {val_test_overlap}")
            
        is_valid = len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0
        
        if is_valid:
            print("✓ Split validation passed - no sample leakage detected")
        else:
            print("✗ Split validation failed - sample leakage detected")
            
        return is_valid 
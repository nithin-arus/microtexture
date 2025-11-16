import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import re
import json
import hashlib
from typing import Tuple, Dict, List, Optional
from datetime import datetime
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
    
    def make_split_key(self, seed: int, split_strategy: str, class_list: List[str], data_hash: str) -> str:
        """
        Create a unique key for identifying splits.
        
        Parameters:
        -----------
        seed : int
            Random seed used for splitting
        split_strategy : str
            Strategy name (e.g., 'sample_aware')
        class_list : List[str]
            List of class names
        data_hash : str
            Hash of data (e.g., hash of filenames)
            
        Returns:
        --------
        str : Split key string
        """
        class_str = ','.join(sorted(class_list))
        key_string = f"{seed}_{split_strategy}_{class_str}_{data_hash}"
        return key_string
    
    def leak_check(self, train_idx: np.ndarray, val_idx: np.ndarray, 
                   test_idx: np.ndarray, sample_ids: np.ndarray) -> Dict:
        """
        Check for data leakage by verifying no sample appears in multiple splits.
        
        Parameters:
        -----------
        train_idx, val_idx, test_idx : np.ndarray
            Indices for each split
        sample_ids : np.ndarray
            Array of sample IDs corresponding to each row index
            
        Returns:
        --------
        dict : Leak check summary with:
            - train_val_overlap: Set of sample IDs overlapping between train and val
            - train_test_overlap: Set of sample IDs overlapping between train and test
            - val_test_overlap: Set of sample IDs overlapping between val and test
            - train_samples: Set of unique sample IDs in train
            - val_samples: Set of unique sample IDs in val
            - test_samples: Set of unique sample IDs in test
            - n_overlaps: Total number of overlapping samples
            - is_valid: Boolean indicating if split is valid (no overlaps)
            
        Raises:
        -------
        AssertionError: If any overlap > 0
        """
        # Extract sample IDs for each split
        train_samples = set(sample_ids[train_idx])
        val_samples = set(sample_ids[val_idx])
        test_samples = set(sample_ids[test_idx])
        
        # Compute overlaps
        train_val_overlap = train_samples & val_samples
        train_test_overlap = train_samples & test_samples
        val_test_overlap = val_samples & test_samples
        
        n_overlaps = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
        is_valid = n_overlaps == 0
        
        summary = {
            'train_val_overlap': list(train_val_overlap),
            'train_test_overlap': list(train_test_overlap),
            'val_test_overlap': list(val_test_overlap),
            'train_samples': list(train_samples),
            'val_samples': list(val_samples),
            'test_samples': list(test_samples),
            'n_train_samples': len(train_samples),
            'n_val_samples': len(val_samples),
            'n_test_samples': len(test_samples),
            'n_overlaps': n_overlaps,
            'is_valid': is_valid
        }
        
        if n_overlaps > 0:
            error_msg = f"Data leakage detected: {n_overlaps} overlapping samples"
            if train_val_overlap:
                error_msg += f"\n  Train-Val overlap: {list(train_val_overlap)[:5]}"
            if train_test_overlap:
                error_msg += f"\n  Train-Test overlap: {list(train_test_overlap)[:5]}"
            if val_test_overlap:
                error_msg += f"\n  Val-Test overlap: {list(val_test_overlap)[:5]}"
            raise AssertionError(error_msg)
        
        return summary
    
    def save_split_indices(self, train_idx: np.ndarray, val_idx: np.ndarray, 
                          test_idx: np.ndarray, out_dir: str, config_hash: str, 
                          seed: int, split_key: str, leak_check_summary: Dict,
                          label_counts: Optional[Dict] = None) -> str:
        """
        Save split indices to disk with metadata.
        
        Parameters:
        -----------
        train_idx, val_idx, test_idx : np.ndarray
            Indices for each split
        out_dir : str
            Output directory to save indices
        config_hash : str
            Hash of configuration
        seed : int
            Random seed used
        split_key : str
            Unique split key
        leak_check_summary : dict
            Leak check results
        label_counts : dict, optional
            Label counts for each split
            
        Returns:
        --------
        str : Path to saved split directory
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Save indices
        np.save(out_path / 'train.npy', train_idx)
        np.save(out_path / 'val.npy', val_idx)
        np.save(out_path / 'test.npy', test_idx)
        
        # Create metadata
        metadata = {
            'seed': seed,
            'config_hash': config_hash,
            'split_key': split_key,
            'split_sizes': {
                'train': len(train_idx),
                'val': len(val_idx),
                'test': len(test_idx)
            },
            'leak_check': leak_check_summary,
            'label_counts': label_counts or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Save metadata
        with open(out_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save leak check summary separately
        with open(out_path / 'leak_check_summary.json', 'w') as f:
            json.dump(leak_check_summary, f, indent=2)
        
        return str(out_path)
    
    def load_split_indices(self, in_dir: str, expected_split_key: Optional[str] = None) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]]:
        """
        Load split indices from disk.
        
        Parameters:
        -----------
        in_dir : str
            Directory containing saved split indices
        expected_split_key : str, optional
            Expected split key to verify. If provided and doesn't match, returns None.
            
        Returns:
        --------
        tuple or None : (train_idx, val_idx, test_idx, metadata_dict) if found and valid, else None
        """
        in_path = Path(in_dir)
        
        if not in_path.exists():
            return None
        
        # Check if files exist
        train_path = in_path / 'train.npy'
        val_path = in_path / 'val.npy'
        test_path = in_path / 'test.npy'
        metadata_path = in_path / 'metadata.json'
        
        if not all([train_path.exists(), val_path.exists(), test_path.exists(), metadata_path.exists()]):
            return None
        
        # Load indices
        train_idx = np.load(train_path)
        val_idx = np.load(val_path)
        test_idx = np.load(test_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Verify split key if provided
        if expected_split_key is not None:
            if metadata.get('split_key') != expected_split_key:
                print(f"Warning: Split key mismatch. Expected: {expected_split_key}, Got: {metadata.get('split_key')}")
                return None
        
        return train_idx, val_idx, test_idx, metadata
    
    def split_by_samples(self, 
                        features_df: pd.DataFrame, 
                        test_size: float = 0.2, 
                        val_size: float = 0.1,
                        stratify_column: str = None,
                        seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
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
        seed : int, optional
            Random seed for splitting. If None, uses self.random_state.
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, Dict] : 
            (train_idx, val_idx, test_idx, leak_check_summary)
            leak_check_summary: dict with overlap counts and sample statistics
        """
        # Use provided seed or default
        split_seed = seed if seed is not None else self.random_state
        
        # Create sample mapping
        sample_mapping = self.create_sample_mapping(features_df)
        
        # Get list of unique samples
        sample_ids_list = list(sample_mapping.keys())
        
        # Create array of sample IDs for each row (for leak checking)
        row_to_sample_id = {}
        for sample_id, indices in sample_mapping.items():
            for idx in indices:
                row_to_sample_id[idx] = sample_id
        sample_ids_array = np.array([row_to_sample_id.get(i, 'unknown') for i in range(len(features_df))])
        
        # Prepare stratification if requested
        if stratify_column and stratify_column in features_df.columns:
            # Create sample-level labels for stratification
            sample_labels = []
            for sample_id in sample_ids_list:
                # Use the most common label for this sample
                sample_indices = sample_mapping[sample_id]
                labels_for_sample = features_df.iloc[sample_indices][stratify_column]
                most_common_label = labels_for_sample.mode().iloc[0] if not labels_for_sample.empty else 0
                sample_labels.append(most_common_label)
            sample_labels = np.array(sample_labels)
        else:
            sample_labels = None
            
        # Split samples (not individual rows) using provided seed
        if sample_labels is not None:
            try:
                train_samples, temp_samples, _, temp_labels = train_test_split(
                    sample_ids_list, sample_labels, 
                    test_size=(test_size + val_size), 
                    random_state=split_seed,
                    stratify=sample_labels
                )
                
                # Further split temp into validation and test
                val_ratio = val_size / (test_size + val_size)
                val_samples, test_samples = train_test_split(
                    temp_samples, 
                    test_size=(1 - val_ratio),
                    random_state=split_seed,
                    stratify=temp_labels
                )
            except ValueError as e:
                print(f"Warning: Stratified splitting failed ({e}), using random splitting")
                train_samples, temp_samples = train_test_split(
                    sample_ids_list, 
                    test_size=(test_size + val_size), 
                    random_state=split_seed
                )
                val_ratio = val_size / (test_size + val_size)
                val_samples, test_samples = train_test_split(
                    temp_samples, 
                    test_size=(1 - val_ratio),
                    random_state=split_seed
                )
        else:
            train_samples, temp_samples = train_test_split(
                sample_ids_list, 
                test_size=(test_size + val_size), 
                random_state=split_seed
            )
            val_ratio = val_size / (test_size + val_size)
            val_samples, test_samples = train_test_split(
                temp_samples, 
                test_size=(1 - val_ratio),
                random_state=split_seed
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
        
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)
        
        # Perform leak check
        leak_check_summary = self.leak_check(train_indices, val_indices, test_indices, sample_ids_array)
        
        print(f"Sample-aware split completed (seed={split_seed}):")
        print(f"  Train: {len(train_samples)} samples ({len(train_indices)} rows)")
        print(f"  Val:   {len(val_samples)} samples ({len(val_indices)} rows)")
        print(f"  Test:  {len(test_samples)} samples ({len(test_indices)} rows)")
        print(f"  Leak check: {'✓ PASSED' if leak_check_summary['is_valid'] else '✗ FAILED'}")
        
        return train_indices, val_indices, test_indices, leak_check_summary
    
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
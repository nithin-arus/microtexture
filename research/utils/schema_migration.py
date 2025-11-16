"""
Feature schema migration utilities.

Checks and fixes feature schemas, removing legacy columns.
"""

import pandas as pd
from typing import Tuple, List, Optional
from analysis.feature_extractor import get_feature_schema


def check_feature_schema(df_columns: List[str], expected_schema: Optional[List[str]] = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Check feature schema against expected schema.
    
    Parameters:
    -----------
    df_columns : List[str]
        Column names from DataFrame
    expected_schema : List[str], optional
        Expected feature schema. If None, uses get_feature_schema().
        
    Returns:
    --------
    tuple : (missing_features, extra_features, legacy_features)
        - missing_features: Features in expected but not in df
        - extra_features: Features in df but not in expected
        - legacy_features: Legacy features to remove (variance, haralick_asm)
    """
    if expected_schema is None:
        expected_schema = get_feature_schema()
    
    # Metadata columns that are allowed
    metadata_cols = ['filename', 'path', 'label', 'fractal_overlay_path', 'fractal_equation',
                     'fractal_hurst_exponent', 'fractal_amplitude_scaling', 'fractal_spectrum_corr',
                     'fractal_spectrum_rmse', 'fractal_fitting_success']
    
    # Legacy features to remove
    legacy_features = ['variance', 'haralick_asm', 'fractal_goodness_of_fit']
    
    # Filter to feature columns only
    df_feature_cols = [col for col in df_columns if col not in metadata_cols]
    expected_set = set(expected_schema)
    df_set = set(df_feature_cols)
    
    # Find missing and extra
    missing_features = sorted(list(expected_set - df_set))
    extra_features = sorted(list(df_set - expected_set))
    
    # Identify legacy features present
    legacy_present = [f for f in legacy_features if f in df_columns]
    
    return missing_features, extra_features, legacy_present


def auto_fix_schema(df: pd.DataFrame, expected_schema: Optional[List[str]] = None,
                   log_migrations: bool = True) -> pd.DataFrame:
    """
    Automatically fix schema by removing legacy columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to fix
    expected_schema : List[str], optional
        Expected feature schema. If None, uses get_feature_schema().
    log_migrations : bool
        Whether to print migration log
        
    Returns:
    --------
    pd.DataFrame : DataFrame with legacy columns removed
    """
    missing, extra, legacy = check_feature_schema(df.columns.tolist(), expected_schema)
    
    if log_migrations:
        if legacy:
            print(f"Migrating schema: Removing legacy features: {legacy}")
        if missing:
            print(f"Warning: Missing expected features: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if extra:
            print(f"Info: Extra features (not in schema): {extra[:10]}{'...' if len(extra) > 10 else ''}")
    
    # Remove legacy columns
    if legacy:
        df = df.drop(columns=legacy, errors='ignore')
        if log_migrations:
            print(f"Removed {len(legacy)} legacy columns")
    
    return df


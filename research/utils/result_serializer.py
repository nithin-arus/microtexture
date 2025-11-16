"""
Result serialization utilities for reproducibility.

Saves results with config hashes, seeds, and environment information.
"""

import json
import pandas as pd
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def serialize_results(results: Dict[str, Any], config: Dict[str, Any], 
                     out_dir: str) -> None:
    """
    Serialize results to JSON and CSV with config hashes and environment info.
    
    Parameters:
    -----------
    results : dict
        Results dictionary (e.g., model results, metrics)
    config : dict
        Configuration dictionary
    out_dir : str
        Output directory to save serialized results
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Create config hash
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    # Get environment info
    try:
        env_freeze = subprocess.check_output(['pip', 'freeze'], 
                                            stderr=subprocess.DEVNULL).decode('utf-8')
    except:
        env_freeze = "Could not capture environment"
    
    # Create serialization dictionary
    serialized = {
        'timestamp': datetime.now().isoformat(),
        'config_hash': config_hash,
        'config': config,
        'results': results,
        'environment': env_freeze
    }
    
    # Save as JSON
    with open(out_path / 'results.json', 'w') as f:
        json.dump(serialized, f, indent=2, default=str)
    
    # Save config separately
    with open(out_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    # Save environment
    with open(out_path / 'environment.txt', 'w') as f:
        f.write(env_freeze)
    
    # If results contain DataFrames, save them as CSV
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                value.to_csv(out_path / f'{key}.csv', index=False)
            elif isinstance(value, dict) and 'aggregated' in value:
                # Multi-seed results
                if isinstance(value['aggregated'], pd.DataFrame):
                    value['aggregated'].to_csv(out_path / f'{key}_aggregated.csv', index=False)


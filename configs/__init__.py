"""
Configuration loader for the project
"""

import yaml
import torch
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Optional path to config file. If None, uses default config.yaml
        
    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update device based on availability
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    
    return config

# Load default configuration
DEFAULT_CONFIG = load_config()
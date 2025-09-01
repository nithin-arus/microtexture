#!/usr/bin/env python3
"""Simple test script to check if Python execution works."""

import os
import sys

print("Python version:", sys.version)
print("Current working directory:", os.getcwd())
print("Files in current directory:")
for file in os.listdir('.'):
    print(f"  {file}")

print("\nTesting basic imports...")
try:
    import numpy as np
    print("✓ numpy imported successfully")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")

try:
    import cv2
    print("✓ opencv imported successfully")
except ImportError as e:
    print(f"✗ opencv import failed: {e}")

try:
    from preprocess.preprocess import preprocess_image
    print("✓ preprocess module imported successfully")
except ImportError as e:
    print(f"✗ preprocess import failed: {e}")

print("\nScript completed successfully!")


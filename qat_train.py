#!/usr/bin/env python3
"""
Wrapper script for QAT training.

This script allows running the QAT training from the root directory
while keeping the source code organized in the src/ folder.

Author: Yin Cao
Date: 2025-08-08
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Import and run the main function
if __name__ == "__main__":
    from src.qat_train import main
    main()

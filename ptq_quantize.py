#!/usr/bin/env python3
"""
Wrapper script for PTQ quantization.

This script allows running the PTQ quantization from the root directory
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
    from src.ptq_quantize import main
    main()

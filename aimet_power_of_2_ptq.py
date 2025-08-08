#!/usr/bin/env python3
"""
Wrapper script for AIMET PTQ quantization.

This script allows running the AIMET PTQ quantization from the root directory
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
    from src.aimet_power_of_2_ptq import main
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test to check if train.py main() can start execution.
"""

import sys
import logging

# Reduce logging overhead
logging.basicConfig(level=logging.WARNING)

print("Attempting to run main()...")
print("=" * 60)

try:
    from train import main
    print("[OK] train.main() imported successfully")
    print("\nStarting execution...")
    print("-" * 60)
    
    # Run main with short timeout
    result = main()
    
    if result:
        print("-" * 60)
        print("[OK] Pipeline completed successfully!")
    else:
        print("-" * 60)
        print("[WARN] Pipeline returned False (check logs)")
        
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

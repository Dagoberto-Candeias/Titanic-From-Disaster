#!/usr/bin/env python
"""
Manual runner for the training pipeline kept for convenience. This script is
intended to be executed interactively and is intentionally *not* a test.
"""
import sys
import logging
from train import main

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("Attempting to run main()...\n" + "=" * 60)
    try:
        res = main()
        if res:
            print("[OK] Pipeline completed successfully!")
        else:
            print("[WARN] Pipeline returned False (check logs)")
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

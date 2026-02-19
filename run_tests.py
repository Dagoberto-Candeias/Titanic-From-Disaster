#!/usr/bin/env python
"""
Test runner that works from any Python environment.
This script runs pytest programmatically on the titanic project.
"""

import subprocess
import sys
import os

def main():
    """Run pytest with proper configuration."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    print("=" * 70)
    print("RUNNING PYTEST ON TITANIC PROJECT")
    print("=" * 70)
    print()
    
    # Run pytest
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"]
    result = subprocess.run(cmd, capture_output=False)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())

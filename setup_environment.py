#!/usr/bin/env python
"""
Helper script to manage the titanic_ml conda environment.
Run this from the conda base environment or PowerShell.
"""
import subprocess
import sys
import os
import platform

def run_command(cmd, shell=False):
    """Run a shell command and return the result."""
    try:
        if isinstance(cmd, str) and not shell:
            cmd = cmd.split()
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)

def check_conda():
    """Check if conda is available."""
    code, out, err = run_command("conda --version")
    return code == 0

def create_env():
    """Create the titanic_ml environment."""
    print("\n" + "="*70)
    print("CREATING TITANIC_ML CONDA ENVIRONMENT")
    print("="*70 + "\n")
    
    code, out, err = run_command("conda env create -f environment.yml", shell=True)
    if code == 0:
        print(f"✓ Environment created successfully\n{out}")
        return True
    else:
        # If env exists, try updating
        print(f"Creating new or updating existing environment...")
        code, out, err = run_command(
            "conda env update -f environment.yml --name titanic_ml",
            shell=True
        )
        print(out)
        if err:
            print(f"Warnings/Errors:\n{err}")
        return code == 0

def activate_and_test():
    """Activate environment and run tests."""
    print("\n" + "="*70)
    print("TESTING ENVIRONMENT")
    print("="*70 + "\n")
    
    if platform.system() == "Windows":
        test_cmd = "conda activate titanic_ml && python test_warnings_check.py"
        full_cmd = f"cmd.exe /c \"{test_cmd}\""
    else:
        test_cmd = "source activate titanic_ml && python test_warnings_check.py"
        full_cmd = test_cmd
    
    code, out, err = run_command(full_cmd, shell=True)
    print(out)
    if err:
        print(f"Errors:\n{err}")
    
    return code == 0

def main():
    """Main setup flow."""
    if not check_conda():
        print("❌ Conda is not available. Please install Anaconda or Miniconda.")
        sys.exit(1)
    
    if create_env():
        print("\n✓ Environment is ready!")
        print("\nNext steps:")
        print("  1. Activate the environment:")
        if platform.system() == "Windows":
            print("     conda activate titanic_ml")
        else:
            print("     source activate titanic_ml")
        print("\n  2. Run tests:")
        print("     pytest -q")
        print("\n  3. Generate report:")
        print("     python src/gerar_relatorio_titanic.py")
    else:
        print("\n❌ Failed to create environment")
        sys.exit(1)

if __name__ == "__main__":
    main()

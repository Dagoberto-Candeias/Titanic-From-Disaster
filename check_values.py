#!/usr/bin/env python3
"""Check for remaining .values usages in the codebase."""

import os
import re
from pathlib import Path

def check_values_usage():
    """Find all .values usages (excluding method calls and comments)."""
    
    # Directories to check
    dirs_to_check = [
        'titanic_pipeline',
        'src',
        'tests'
    ]
    
    # Pattern to match .values but NOT .values()
    # This uses negative lookahead to exclude method calls
    pattern = re.compile(r'\.values(?!\s*\()')
    
    findings = []
    
    for root_dir in dirs_to_check:
        if not os.path.isdir(root_dir):
            continue
            
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if not file.endswith('.py'):
                    continue
                    
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            # Skip comments and docstrings
                            if line.strip().startswith('#'):
                                continue
                            
                            # Find matches
                            for match in pattern.finditer(line):
                                # Skip if in a comment
                                comment_idx = line.find('#')
                                if comment_idx >= 0 and match.start() > comment_idx:
                                    continue
                                    
                                findings.append({
                                    'file': filepath,
                                    'line': line_num,
                                    'content': line.strip()
                                })
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    # Print results
    if findings:
        print(f"Found {len(findings)} instances of .values:")
        print("-" * 80)
        for item in findings:
            print(f"{item['file']}:{item['line']}")
            print(f"  {item['content']}")
            print()
    else:
        print("✓ No remaining .values usages found (excluding method calls)")
    
    return len(findings) == 0

if __name__ == '__main__':
    success = check_values_usage()
    exit(0 if success else 1)

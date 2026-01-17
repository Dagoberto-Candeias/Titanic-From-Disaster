#!/bin/bash
echo "Starting auto-commit watcher..."
export SKIP=pytest
python auto_commit.py

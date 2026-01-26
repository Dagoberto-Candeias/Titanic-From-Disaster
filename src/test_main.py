#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test to check if train.py main() can start execution.
"""

import sys
import logging

# Reduce logging overhead
logging.basicConfig(level=logging.WARNING)

def test_train_importable():
    """Simple test: train module should be importable without side-effects."""
    import importlib

    # Import the train module to ensure it doesn't execute heavy work on import
    importlib.import_module("train")


# The legacy manual runner was moved to `scripts/manual_train_runner.py` to
# avoid executing the full pipeline during pytest collection. Tests should
# only import the module and assert importability without side-effects.

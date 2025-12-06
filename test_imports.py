#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify imports work correctly.
"""

print("1. Testing standard imports...")
import os
import pandas as pd
import numpy as np
print("   ✅ Standard imports OK")

print("2. Testing titanic_pipeline imports...")
try:
    from titanic_pipeline.preprocessing import AdvancedFeatureEngineer
    print("   ✅ AdvancedFeatureEngineer imported")
except ImportError as e:
    print(f"   ⚠️  Could not import AdvancedFeatureEngineer: {e}")

try:
    from titanic_pipeline.utils import ensure_feature_cols_intersection
    print("   ✅ ensure_feature_cols_intersection imported")
except ImportError as e:
    print(f"   ⚠️  Could not import ensure_feature_cols_intersection: {e}")

try:
    from titanic_pipeline.core.utils import get_cache_key, cache_result, load_cached_result
    print("   ✅ Core utils imported")
except ImportError as e:
    print(f"   ⚠️  Could not import core utils: {e}")

print("3. Testing train.py setup imports...")
try:
    from sklearn.ensemble import VotingClassifier
    from sklearn.model_selection import cross_val_score
    print("   ✅ Sklearn imports OK")
except ImportError as e:
    print(f"   ⚠️  Sklearn import failed: {e}")

print("\n✅ ALL IMPORT TESTS COMPLETED")
print("\nTo run the full pipeline, use: python train.py")

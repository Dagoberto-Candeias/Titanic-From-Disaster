# TODO: Titanic Pipeline Comprehensive Improvements

## Phase 1: Critical Fixes & Imports (High Priority)
- [x] Fix `set_global_seeds` import error
- [x] Fix any remaining circular imports
- [x] Resolve critical bugs in data processing
- [x] Improve error handling and validation
- [x] Add missing dependencies checks

## Phase 2: Performance Improvements (High Priority)
- [ ] Integrate ModelingManager into main script for parallel training with timeouts
- [ ] Optimize slow models (SVM, KNN) with better defaults and early stopping
- [ ] Implement model training timeouts to prevent hanging
- [ ] Add progress bars for long operations
- [ ] Optimize feature engineering bottlenecks

## Phase 3: Memory Optimizations (High Priority)
- [ ] Add memory monitoring and reporting
- [ ] Implement garbage collection triggers
- [ ] Optimize data types further for large datasets
- [ ] Add memory-efficient feature engineering options
- [ ] Implement chunked processing for large datasets

## Phase 4: Configuration Enhancements (Medium Priority)
- [ ] Move all hardcoded parameters to config.py
- [ ] Add configuration validation and schema checking
- [ ] Make feature engineering parameters configurable
- [ ] Add environment-specific configurations
- [ ] Implement configuration hot-reloading

## Phase 5: SHAP & Explainability (Medium Priority)
- [ ] Add fallback explainability methods for non-tree models
- [ ] Implement LIME or other alternatives when SHAP fails
- [ ] Add model-agnostic explainability options
- [ ] Improve SHAP error handling and logging

## Phase 6: Comprehensive Testing (Critical)
- [ ] Expand run_unit_tests() with more comprehensive tests
- [ ] Implement integration tests for full pipeline
- [ ] Add performance regression tests
- [ ] Test edge cases (missing data, outliers, different sizes)
- [ ] Add smoke tests for critical paths
- [ ] Implement automated testing pipeline

## Phase 7: Additional Improvements (Low Priority)
- [ ] Add data quality checks and anomaly detection
- [ ] Implement model versioning and comparison
- [ ] Add experiment tracking
- [ ] Improve documentation and code quality
- [ ] Add CI/CD pipeline support

## Status
- ✅ Import errors fixed (set_global_seeds)
- ✅ Basic pipeline execution verified
- 🔄 In progress: Performance and memory optimizations
- 📋 Planned: Configuration, testing, and explainability improvements

## Metrics Targets
- Performance: 30-50% faster execution
- Memory: 20-40% reduction in peak usage
- Reliability: 90%+ test coverage
- Maintainability: Fully configurable and well-tested

## Current Implementation Plan (Phase 2 & 6 Priority)
### Task 1: Integrate ModelingManager
- [ ] Modify main script to use ModelingManager.train_all_models() instead of manual ProcessPoolExecutor
- [ ] Update imports to include ModelingManager
- [ ] Test parallel training with timeouts

### Task 2: Optimize Slow Models
- [ ] Update get_base_models() to include optimized parameters for SVC, KNN, MLP
- [ ] Add early stopping where possible
- [ ] Test model training times

### Task 3: Expand Unit Tests
- [ ] Add tests for data validation, preprocessing, feature engineering
- [ ] Add tests for model training and ensemble creation
- [ ] Add tests for reporting and file generation

### Task 4: Add Integration Tests
- [ ] Create run_integration_tests() function
- [ ] Test full pipeline execution
- [ ] Test with different configurations

### Task 5: Add Progress Monitoring
- [ ] Add tqdm progress bars for model training
- [ ] Add memory usage monitoring
- [ ] Add timing reports for each phase

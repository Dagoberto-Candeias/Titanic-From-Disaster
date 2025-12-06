"""
Data validation utilities for Titanic ML Pipeline.
"""

import json
import logging
import os
from typing import Dict, List, Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data schema and integrity."""

    def __init__(self, expected_schemas: Dict[str, Dict[str, str]]):
        self.expected_schemas = expected_schemas

    def validate_data_schema(
        self,
        df: pd.DataFrame,
        expected_columns: List[str],
        dataset_name: str,
        output_dir: str = "output/relatorios"
    ) -> bool:
        """
        Validate DataFrame schema against expected schema.

        Args:
            df: DataFrame to validate
            expected_columns: Expected column names
            dataset_name: Name of the dataset for logging
            output_dir: Directory to save validation report

        Returns:
            True if validation passes, False otherwise
        """
        os.makedirs(output_dir, exist_ok=True)

        schema_report = {
            "dataset": dataset_name,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "expected_columns": expected_columns,
            "actual_columns": list(df.columns),
            "missing_columns": [],
            "extra_columns": [],
            "dtype_mismatches": [],
            "validation_passed": True,
        }

        # Check for missing columns
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            schema_report["missing_columns"] = list(missing_cols)
            schema_report["validation_passed"] = False
            logger.error(f"❌ Missing columns in {dataset_name}: {missing_cols}")

        # Check for extra columns
        extra_cols = set(df.columns) - set(expected_columns)
        if extra_cols:
            schema_report["extra_columns"] = list(extra_cols)
            logger.warning(f"⚠️  Extra columns in {dataset_name}: {extra_cols}")

        # Check data types if schema provided
        if dataset_name in self.expected_schemas:
            expected_schema = self.expected_schemas[dataset_name]
            for col, expected_dtype in expected_schema.items():
                if col in df.columns:
                    actual_dtype = str(df[col].dtype)
                    if actual_dtype != expected_dtype:
                        schema_report["dtype_mismatches"].append({
                            "column": col,
                            "expected": expected_dtype,
                            "actual": actual_dtype
                        })
                        logger.warning(
                            f"⚠️  Dtype mismatch in {dataset_name}.{col}: "
                            f"expected {expected_dtype}, got {actual_dtype}"
                        )

        # Save validation report
        report_path = os.path.join(output_dir, f"schema_validation_{dataset_name}.json")
        with open(report_path, "w") as f:
            json.dump(schema_report, f, indent=2, default=str)

        if schema_report["validation_passed"]:
            logger.info(f"✅ Schema validation passed for {dataset_name}")
        else:
            logger.error(f"❌ Schema validation failed for {dataset_name}")

        return schema_report["validation_passed"]

    def validate_data_integrity(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Perform basic data integrity checks.

        Args:
            df: DataFrame to check
            dataset_name: Name of the dataset

        Returns:
            Dictionary with integrity check results
        """
        integrity_report = {
            "dataset": dataset_name,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "duplicate_rows": df.duplicated().sum(),
            "missing_values": df.isnull().sum().to_dict(),
            "infinite_values": {},
            "zero_variance_columns": [],
            "integrity_passed": True,
        }

        # Check for infinite values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                integrity_report["infinite_values"][col] = inf_count
                integrity_report["integrity_passed"] = False
                logger.warning(f"⚠️  Infinite values in {dataset_name}.{col}: {inf_count}")

        # Check for zero variance columns
        for col in numeric_cols:
            if df[col].var() == 0:
                integrity_report["zero_variance_columns"].append(col)
                logger.warning(f"⚠️  Zero variance column in {dataset_name}: {col}")

        # Check for duplicate rows
        if integrity_report["duplicate_rows"] > 0:
            logger.warning(
                f"⚠️  Duplicate rows in {dataset_name}: {integrity_report['duplicate_rows']}"
            )

        return integrity_report

    def ensure_feature_cols_intersection(
        self,
        train_cols: List[str],
        test_cols: List[str],
        feature_cols: List[str]
    ) -> List[str]:
        """
        Ensure feature columns exist in both train and test sets.

        Args:
            train_cols: Training column names
            test_cols: Test column names
            feature_cols: Desired feature columns

        Returns:
            Filtered feature columns that exist in both datasets
        """
        train_set = set(train_cols)
        test_set = set(test_cols)
        common_cols = train_set & test_set
        filtered_cols = [col for col in feature_cols if col in common_cols]

        removed_cols = set(feature_cols) - set(filtered_cols)
        if removed_cols:
            logger.warning(f"⚠️  Removed features not in both datasets: {removed_cols}")

        return filtered_cols

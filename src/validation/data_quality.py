"""
Great Expectations data validation suite.
Automated data quality checks and profiling.
"""

from great_expectations.dataset import Dataset
from great_expectations.dataset.pandas_dataset import PandasDataset
import pandas as pd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityLogDataValidator:
    """
    Data quality validation for security logs using Great Expectations
    """
    
    def __init__(self):
        self.expectations = []
        
    def validate_raw_logs(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validates raw security log dataframe
        
        Returns:
            (is_valid, validation_report)
        """
        
        dataset = PandasDataset(df)
        results = []
        
        # Column existence
        required_columns = [
            "timestamp", "user_id", "ip_address", "path", 
            "method", "request", "status_code", "latency_ms"
        ]
        
        for col in required_columns:
            result = dataset.expect_column_to_exist(col)
            results.append(result)
            logger.info(f"Column existence check '{col}': {result['success']}")
        
        # Type validation
        type_checks = {
            "timestamp": "object",  # datetime
            "status_code": "int64",
            "latency_ms": "int64"
        }
        
        for col, expected_type in type_checks.items():
            result = dataset.expect_column_values_to_be_of_type(
                col, expected_type
            )
            results.append(result)
            logger.info(f"Type check '{col}' ({expected_type}): {result['success']}")
        
        # Value range validation
        result = dataset.expect_column_values_to_be_between(
            "status_code", min_value=100, max_value=599
        )
        results.append(result)
        logger.info(f"Status code range check: {result['success']}")
        
        result = dataset.expect_column_values_to_be_between(
            "latency_ms", min_value=0, max_value=60000
        )
        results.append(result)
        logger.info(f"Latency range check: {result['success']}")
        
        # Uniqueness checks
        result = dataset.expect_column_values_to_be_unique("user_id")
        logger.info(f"User ID uniqueness: {result['success']}")
        
        # Null value checks
        for col in required_columns:
            result = dataset.expect_column_values_to_not_be_null(col)
            results.append(result)
            if not result['success']:
                logger.warning(f"Null values found in '{col}'")
        
        # Regex validation for IP addresses
        result = dataset.expect_column_values_to_match_regex(
            "ip_address",
            regex=r"^(\d{1,3}\.){3}\d{1,3}$"
        )
        results.append(result)
        logger.info(f"IP address format check: {result['success']}")
        
        # HTTP method validation
        result = dataset.expect_column_values_to_be_in_set(
            "method", ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
        )
        results.append(result)
        logger.info(f"HTTP method validation: {result['success']}")
        
        all_passed = all(r['success'] for r in results)
        
        return all_passed, {
            "total_checks": len(results),
            "passed": sum(1 for r in results if r['success']),
            "failed": sum(1 for r in results if not r['success']),
            "results": results
        }
    
    
    def validate_features(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validates engineered features dataframe
        """
        
        dataset = PandasDataset(df)
        results = []
        
        # Required feature columns
        feature_columns = [
            "request_length", "has_sql_keywords", "is_admin_path",
            "path_rarity", "method_entropy", "sql_keyword_rarity",
            "path_transition_risk", "repeated_request_count", "method_transition_flag"
        ]
        
        for col in feature_columns:
            result = dataset.expect_column_to_exist(col)
            results.append(result)
        
        # Binary features validation
        binary_features = [
            "has_sql_keywords", "is_admin_path", "method_transition_flag"
        ]
        
        for col in binary_features:
            result = dataset.expect_column_values_to_be_in_set(col, [0, 1])
            results.append(result)
            logger.info(f"Binary validation '{col}': {result['success']}")
        
        # Probability/score features (0-1 range)
        prob_features = ["path_rarity", "sql_keyword_rarity", "path_transition_risk"]
        
        for col in prob_features:
            result = dataset.expect_column_values_to_be_between(
                col, min_value=0, max_value=1
            )
            results.append(result)
            logger.info(f"Probability range '{col}': {result['success']}")
        
        # Integer features validation
        result = dataset.expect_column_values_to_be_of_type("request_length", "int64")
        results.append(result)
        
        result = dataset.expect_column_values_to_be_between(
            "request_length", min_value=0, max_value=5000
        )
        results.append(result)
        logger.info(f"Request length range: {result['success']}")
        
        result = dataset.expect_column_values_to_be_between(
            "repeated_request_count", min_value=0, max_value=1000
        )
        results.append(result)
        logger.info(f"Repeated request count range: {result['success']}")
        
        # No infinite or NaN values
        for col in feature_columns:
            result = dataset.expect_column_values_to_not_be_nan(col)
            results.append(result)
            result = dataset.expect_column_values_to_not_be_null(col)
            results.append(result)
        
        # Label validation (if present)
        if "label" in df.columns:
            result = dataset.expect_column_values_to_be_in_set("label", [0, 1])
            results.append(result)
            logger.info(f"Label validation: {result['success']}")
            
            # Check class balance
            label_counts = df["label"].value_counts()
            class_imbalance = label_counts.max() / label_counts.min() if len(label_counts) == 2 else None
            if class_imbalance and class_imbalance > 10:
                logger.warning(f"High class imbalance detected: {class_imbalance:.2f}x")
        
        all_passed = all(r['success'] for r in results)
        
        return all_passed, {
            "total_checks": len(results),
            "passed": sum(1 for r in results if r['success']),
            "failed": sum(1 for r in results if not r['success']),
            "rows": len(df),
            "columns": len(df.columns),
            "results": results
        }
    
    
    def validate_predictions(self, predictions_df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validates prediction output dataframe
        """
        
        dataset = PandasDataset(predictions_df)
        results = []
        
        # Required prediction columns
        pred_columns = [
            "risk_score", "classifier_prob", "anomaly_score", "context_risk",
            "alert_level", "model_version", "model_stage"
        ]
        
        for col in pred_columns:
            result = dataset.expect_column_to_exist(col)
            results.append(result)
        
        # Score validation (0-1 range)
        score_columns = ["risk_score", "classifier_prob", "anomaly_score", "context_risk"]
        
        for col in score_columns:
            result = dataset.expect_column_values_to_be_between(
                col, min_value=0, max_value=1
            )
            results.append(result)
            logger.info(f"Score range '{col}': {result['success']}")
        
        # Alert level validation
        result = dataset.expect_column_values_to_be_in_set(
            "alert_level", ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        )
        results.append(result)
        logger.info(f"Alert level validation: {result['success']}")
        
        # No nulls in critical columns
        for col in score_columns:
            result = dataset.expect_column_values_to_not_be_null(col)
            results.append(result)
        
        all_passed = all(r['success'] for r in results)
        
        return all_passed, {
            "total_checks": len(results),
            "passed": sum(1 for r in results if r['success']),
            "failed": sum(1 for r in results if not r['success']),
            "predictions": len(predictions_df),
            "results": results
        }
    
    
    def generate_data_profile(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Dict:
        """
        Generates statistical profile of dataset
        """
        
        profile = {
            "dataset_name": dataset_name,
            "shape": df.shape,
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "column_info": {},
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percent": (df.isnull().sum() / len(df) * 100).to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_stats": df.describe().to_dict(),
        }
        
        # Additional categorical info
        for col in df.select_dtypes(include=['object']).columns:
            profile["column_info"][col] = {
                "unique_values": df[col].nunique(),
                "top_values": df[col].value_counts().head(5).to_dict()
            }
        
        return profile


# Utility function for validation in pipelines
def validate_data_quality(df: pd.DataFrame, stage: str = "features") -> bool:
    """
    Quick validation check for common stages
    
    Parameters:
        df: DataFrame to validate
        stage: 'logs', 'features', or 'predictions'
    
    Returns:
        True if validation passed
    """
    
    validator = SecurityLogDataValidator()
    
    if stage == "logs":
        is_valid, report = validator.validate_raw_logs(df)
    elif stage == "features":
        is_valid, report = validator.validate_features(df)
    elif stage == "predictions":
        is_valid, report = validator.validate_predictions(df)
    else:
        raise ValueError(f"Unknown validation stage: {stage}")
    
    if not is_valid:
        logger.error(f"Validation failed at {stage} stage: {report}")
        raise ValueError(f"Data quality check failed for {stage}")
    
    logger.info(f"✅ {stage} validation passed ({report['passed']}/{report['total_checks']} checks)")
    return True

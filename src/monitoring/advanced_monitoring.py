"""
Advanced monitoring using Evidently AI, SHAP explainability,
and real-time feature importance tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestNumberOfMissingValues,
    TestMeanInNRanges,
    TestOutOfRangeValues,
)
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import shap
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedDriftDetector:
    """
    Multi-modal drift detection using statistical tests and ML-based approaches
    """
    
    def __init__(self):
        self.reference_data = None
        self.drift_thresholds = {}
    
    
    def generate_data_drift_report(self, current_data: pd.DataFrame,
                                   reference_data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data drift report using Evidently AI
        """
        
        logger.info("📊 Generating data drift report...")
        
        # Create drift report
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(reference_data=reference_data, current_data=current_data)
        
        report_dict = drift_report.as_dict()
        
        # Extract key metrics
        drift_results = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "report": report_dict,
            "drifted_features": [],
            "drift_score": 0.0,
        }
        
        logger.info("✅ Data drift report generated")
        
        return drift_results
    
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate data quality metrics using Evidently AI
        """
        
        logger.info("🔍 Generating data quality report...")
        
        quality_report = Report(metrics=[DataQualityPreset()])
        quality_report.run(reference_data=df, current_data=df)
        
        report_dict = quality_report.as_dict()
        
        quality_results = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "report": report_dict,
            "missing_values": {},
            "quality_score": 0.0,
        }
        
        logger.info("✅ Data quality report generated")
        
        return quality_results
    
    
    def detect_concept_drift(self, y_true: np.ndarray, y_pred: np.ndarray,
                           window_size: int = 100) -> Dict:
        """
        Detect concept drift using model performance degradation
        over sliding windows
        """
        
        logger.info(f"🎯 Detecting concept drift (window_size={window_size})...")
        
        from sklearn.metrics import accuracy_score
        
        accuracies = []
        windows = []
        
        for i in range(0, len(y_true) - window_size, window_size // 2):
            window_true = y_true[i:i+window_size]
            window_pred = y_pred[i:i+window_size]
            
            acc = accuracy_score(window_true, window_pred)
            accuracies.append(acc)
            windows.append(i)
        
        # Calculate drift (decreasing accuracy trend)
        if len(accuracies) > 1:
            accuracy_slope = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
            concept_drift_detected = accuracy_slope < -0.01  # Negative trend
        else:
            accuracy_slope = 0
            concept_drift_detected = False
        
        drift_results = {
            "concept_drift_detected": concept_drift_detected,
            "accuracy_trend_slope": float(accuracy_slope),
            "accuracies": accuracies,
            "windows": windows,
        }
        
        logger.info(f"{'🚨 Concept drift detected!' if concept_drift_detected else '✅ No concept drift'}")
        
        return drift_results


class SHAPExplainer:
    """
    SHAP-based model explainability for feature importance and prediction explanations
    """
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    
    def initialize_explainer(self, X_background: pd.DataFrame, explainer_type: str = 'tree'):
        """
        Initialize SHAP explainer
        
        Parameters:
            X_background: Background data for SHAP
            explainer_type: 'tree', 'kernel', or 'linear'
        """
        
        logger.info(f"🔧 Initializing SHAP {explainer_type} explainer...")
        
        if explainer_type == 'tree':
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except:
                logger.warning("TreeExplainer not available, falling back to KernelExplainer")
                explainer_type = 'kernel'
        
        if explainer_type == 'kernel':
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                shap.sample(X_background.values, 100)
            )
        elif explainer_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model, X_background)
        
        logger.info("✅ SHAP explainer initialized")
    
    
    def get_global_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate global feature importance using SHAP values
        """
        
        logger.info("🌍 Calculating global feature importance...")
        
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")
        
        shap_values = self.explainer.shap_values(X.values)
        
        # Handle multi-class (take positive class)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Mean absolute SHAP values
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': feature_importance
        }).sort_values('shap_importance', ascending=False)
        
        logger.info("\n🏆 Top 10 features by SHAP importance:")
        logger.info(importance_df.head(10).to_string(index=False))
        
        return importance_df
    
    
    def explain_instance(self, X: pd.DataFrame, instance_idx: int) -> Dict:
        """
        Generate explanation for a single prediction
        """
        
        logger.info(f"🔍 Explaining instance {instance_idx}...")
        
        if self.explainer is None:
            raise ValueError("Explainer not initialized.")
        
        instance = X.iloc[instance_idx:instance_idx+1].values
        shap_values = self.explainer.shap_values(instance)
        
        # Handle multi-class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get prediction
        prediction = self.model.predict_proba(instance)[0]
        
        # Create explanation
        explanation = {
            "instance_idx": instance_idx,
            "prediction_proba": float(prediction[1]),  # Positive class probability
            "contributing_features": {}
        }
        
        # Sort features by absolute SHAP value
        feature_shap = list(zip(self.feature_names, shap_values[0]))
        feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for feature, shap_val in feature_shap[:5]:  # Top 5 features
            explanation["contributing_features"][feature] = {
                "shap_value": float(shap_val),
                "feature_value": float(X[feature].iloc[instance_idx]),
            }
        
        logger.info(f"✅ Instance explanation generated")
        
        return explanation
    
    
    def create_summary_plot_data(self, X: pd.DataFrame) -> Dict:
        """
        Generate data for SHAP summary plot (feature importance visualization)
        """
        
        logger.info("📈 Creating SHAP summary plot data...")
        
        if self.explainer is None:
            raise ValueError("Explainer not initialized.")
        
        shap_values = self.explainer.shap_values(X.values)
        
        # Handle multi-class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Prepare plot data
        plot_data = {
            "features": self.feature_names,
            "shap_values": shap_values.tolist(),
            "feature_values": X.values.tolist(),
        }
        
        logger.info("✅ Summary plot data created")
        
        return plot_data


class RealTimeFeatureMonitor:
    """
    Track feature statistics in real-time for anomaly detection
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.feature_stats = {}
        self.alerts = []
    
    
    def update_statistics(self, df: pd.DataFrame):
        """
        Update rolling statistics for features
        """
        
        logger.info("📊 Updating feature statistics...")
        
        for col in df.select_dtypes(include=[np.number]).columns:
            self.feature_stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "q25": df[col].quantile(0.25),
                "q75": df[col].quantile(0.75),
                "count": len(df),
            }
    
    
    def detect_anomalies(self, new_data: pd.DataFrame, z_score_threshold: float = 3.0) -> Dict:
        """
        Detect anomalous features using z-score
        """
        
        logger.info(f"🚨 Detecting feature anomalies (z-threshold={z_score_threshold})...")
        
        anomalies = {}
        
        for col in new_data.select_dtypes(include=[np.number]).columns:
            if col not in self.feature_stats:
                continue
            
            stats = self.feature_stats[col]
            mean = stats["mean"]
            std = stats["std"]
            
            if std < 1e-6:  # Constant feature
                continue
            
            z_scores = np.abs((new_data[col] - mean) / std)
            anomalous_indices = np.where(z_scores > z_score_threshold)[0]
            
            if len(anomalous_indices) > 0:
                anomalies[col] = {
                    "count": len(anomalous_indices),
                    "indices": anomalous_indices.tolist(),
                    "values": new_data[col].iloc[anomalous_indices].tolist(),
                    "z_scores": z_scores[anomalous_indices].tolist(),
                }
                logger.warning(f"  ⚠️  {col}: {len(anomalous_indices)} anomalies detected")
        
        return anomalies


class PredictionMonitor:
    """
    Monitor prediction distribution and model performance degradation
    """
    
    def __init__(self):
        self.prediction_history = []
        self.performance_history = []
    
    
    def track_predictions(self, predictions: np.ndarray, y_true: Optional[np.ndarray] = None):
        """
        Track prediction statistics
        """
        
        logger.info("📊 Tracking predictions...")
        
        record = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "mean_prediction": float(predictions.mean()),
            "std_prediction": float(predictions.std()),
            "min_prediction": float(predictions.min()),
            "max_prediction": float(predictions.max()),
        }
        
        if y_true is not None:
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_true, (predictions > 0.5).astype(int))
            record["accuracy"] = float(accuracy)
        
        self.prediction_history.append(record)
        
        logger.info(f"✅ Recorded prediction statistics: mean={record['mean_prediction']:.4f}")
    
    
    def detect_performance_degradation(self, lookback_window: int = 10) -> Dict:
        """
        Detect if model performance is degrading
        """
        
        logger.info("📉 Checking for performance degradation...")
        
        if len(self.prediction_history) < lookback_window:
            return {"degradation_detected": False, "reason": "Insufficient history"}
        
        recent_accuracies = [
            record.get("accuracy", 0.5) 
            for record in self.prediction_history[-lookback_window:]
            if "accuracy" in record
        ]
        
        if len(recent_accuracies) < 2:
            return {"degradation_detected": False, "reason": "No accuracy records"}
        
        # Check for downward trend
        accuracy_trend = np.polyfit(range(len(recent_accuracies)), recent_accuracies, 1)[0]
        degradation_detected = accuracy_trend < -0.02  # Losing >2% per step
        
        result = {
            "degradation_detected": degradation_detected,
            "accuracy_trend": float(accuracy_trend),
            "recent_accuracies": recent_accuracies,
            "recommendation": "Consider retraining" if degradation_detected else "Model performing normally"
        }
        
        if degradation_detected:
            logger.warning(f"🚨 Performance degradation detected: {accuracy_trend:.4f} per step")
        
        return result


# Utility function for comprehensive monitoring
def run_comprehensive_monitoring(
    current_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    model,
    feature_names: List[str],
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
) -> Dict:
    """
    Run all monitoring checks and return comprehensive report
    """
    
    logger.info("🚀 Running comprehensive monitoring...")
    
    # Drift detection
    drift_detector = AdvancedDriftDetector()
    drift_report = drift_detector.generate_data_drift_report(current_data, reference_data)
    quality_report = drift_detector.generate_data_quality_report(current_data)
    
    # SHAP explainability
    shap_explainer = SHAPExplainer(model, feature_names)
    shap_explainer.initialize_explainer(reference_data, explainer_type='tree')
    feature_importance = shap_explainer.get_global_feature_importance(current_data)
    
    # Feature monitoring
    feature_monitor = RealTimeFeatureMonitor()
    feature_monitor.update_statistics(reference_data)
    feature_anomalies = feature_monitor.detect_anomalies(current_data)
    
    # Prediction monitoring
    pred_monitor = PredictionMonitor()
    if y_pred is not None:
        pred_monitor.track_predictions(y_pred, y_true)
        perf_degradation = pred_monitor.detect_performance_degradation()
    else:
        perf_degradation = {"degradation_detected": False}
    
    monitoring_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "drift_report": drift_report,
        "quality_report": quality_report,
        "feature_importance": feature_importance.to_dict('records'),
        "feature_anomalies": feature_anomalies,
        "performance_degradation": perf_degradation,
    }
    
    logger.info("✅ Comprehensive monitoring complete")
    
    return monitoring_report

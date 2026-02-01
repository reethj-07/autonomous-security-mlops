"""
A/B Testing Framework for security model experimentation
Statistical hypothesis testing and experiment tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy import stats
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for A/B test"""
    experiment_id: str
    variant_a_name: str
    variant_b_name: str
    metric: str  # 'precision', 'recall', 'f1', 'roc_auc', etc.
    alpha: float = 0.05  # Significance level
    power: float = 0.8  # Statistical power
    expected_effect_size: float = 0.05  # Minimum detectable effect


class StatisticalTester:
    """
    Performs statistical hypothesis testing for A/B experiments
    """
    
    def __init__(self):
        self.test_results = {}
    
    
    def calculate_sample_size(self, baseline_metric: float, effect_size: float,
                             alpha: float = 0.05, power: float = 0.8) -> int:
        """
        Calculate required sample size using Cochran's formula
        
        Parameters:
            baseline_metric: Baseline performance (e.g., 0.85 for 85% accuracy)
            effect_size: Minimum detectable effect (e.g., 0.05 for 5% improvement)
            alpha: Type I error rate
            power: Statistical power (1 - Type II error)
        
        Returns:
            Required sample size per variant
        """
        
        from scipy.stats import norm
        
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        
        p1 = baseline_metric
        p2 = baseline_metric + effect_size
        p_pooled = (p1 + p2) / 2
        
        # Cochran's formula
        n = (z_alpha + z_beta) ** 2 * (p_pooled * (1 - p_pooled)) / ((p1 - p2) ** 2)
        
        logger.info(f"📊 Required sample size: {int(np.ceil(n))} per variant")
        
        return int(np.ceil(n))
    
    
    def two_proportion_ztest(self, variant_a_success: int, variant_a_total: int,
                            variant_b_success: int, variant_b_total: int,
                            alpha: float = 0.05) -> Dict:
        """
        Two-proportion z-test for binary metrics
        
        Parameters:
            variant_a_success: Number of successes in variant A
            variant_a_total: Total samples in variant A
            variant_b_success: Number of successes in variant B
            variant_b_total: Total samples in variant B
        
        Returns:
            Test results dictionary
        """
        
        from statsmodels.stats.proportion import proportions_ztest
        
        # Calculate proportions
        p_a = variant_a_success / variant_a_total
        p_b = variant_b_success / variant_b_total
        effect_size = p_b - p_a
        
        # Perform z-test
        count = np.array([variant_a_success, variant_b_success])
        nobs = np.array([variant_a_total, variant_b_total])
        
        z_stat, p_value = proportions_ztest(count, nobs)
        
        # Confidence interval
        se = np.sqrt((p_a * (1 - p_a) / variant_a_total) + (p_b * (1 - p_b) / variant_b_total))
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_lower = effect_size - z_crit * se
        ci_upper = effect_size + z_crit * se
        
        significant = p_value < alpha
        
        result = {
            "test_type": "two_proportion_ztest",
            "variant_a_rate": float(p_a),
            "variant_b_rate": float(p_b),
            "effect_size": float(effect_size),
            "effect_size_percent": float(effect_size * 100),
            "z_statistic": float(z_stat),
            "p_value": float(p_value),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "significant": bool(significant),
            "alpha": alpha,
        }
        
        return result
    
    
    def two_sample_ttest(self, variant_a_scores: np.ndarray,
                        variant_b_scores: np.ndarray,
                        alpha: float = 0.05) -> Dict:
        """
        Two-sample t-test for continuous metrics (e.g., F1 score distributions)
        
        Parameters:
            variant_a_scores: Array of metric scores for variant A
            variant_b_scores: Array of metric scores for variant B
        
        Returns:
            Test results dictionary
        """
        
        mean_a = np.mean(variant_a_scores)
        mean_b = np.mean(variant_b_scores)
        effect_size = mean_b - mean_a
        
        # Welch's t-test (doesn't assume equal variances)
        t_stat, p_value = stats.ttest_ind(variant_a_scores, variant_b_scores, equal_var=False)
        
        # Cohen's d (standardized effect size)
        pooled_std = np.sqrt(
            ((len(variant_a_scores) - 1) * np.var(variant_a_scores, ddof=1) +
             (len(variant_b_scores) - 1) * np.var(variant_b_scores, ddof=1)) /
            (len(variant_a_scores) + len(variant_b_scores) - 2)
        )
        cohens_d = effect_size / pooled_std if pooled_std > 0 else 0
        
        significant = p_value < alpha
        
        result = {
            "test_type": "two_sample_ttest",
            "variant_a_mean": float(mean_a),
            "variant_b_mean": float(mean_b),
            "effect_size": float(effect_size),
            "cohens_d": float(cohens_d),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "variant_a_std": float(np.std(variant_a_scores)),
            "variant_b_std": float(np.std(variant_b_scores)),
            "significant": bool(significant),
            "alpha": alpha,
        }
        
        return result
    
    
    def mann_whitney_utest(self, variant_a_scores: np.ndarray,
                          variant_b_scores: np.ndarray,
                          alpha: float = 0.05) -> Dict:
        """
        Non-parametric Mann-Whitney U test (robust to non-normal distributions)
        """
        
        median_a = np.median(variant_a_scores)
        median_b = np.median(variant_b_scores)
        
        u_stat, p_value = stats.mannwhitneyu(variant_a_scores, variant_b_scores, alternative='two-sided')
        
        significant = p_value < alpha
        
        result = {
            "test_type": "mann_whitney_utest",
            "variant_a_median": float(median_a),
            "variant_b_median": float(median_b),
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "significant": bool(significant),
            "alpha": alpha,
        }
        
        return result


class ABTestExperiment:
    """
    Manages end-to-end A/B testing experiment
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        self.tester = StatisticalTester()
        self.created_at = datetime.utcnow()
    
    
    def run_experiment(self, variant_a_data: pd.DataFrame, variant_a_model,
                      variant_b_data: pd.DataFrame, variant_b_model,
                      y_true: np.ndarray) -> Dict:
        """
        Run full A/B test comparing two model variants
        
        Parameters:
            variant_a_data: Feature data for variant A
            variant_a_model: Trained model for variant A
            variant_b_data: Feature data for variant B
            variant_b_model: Trained model for variant B
            y_true: Ground truth labels
        
        Returns:
            Comprehensive test results
        """
        
        logger.info(f"🚀 Running A/B test: {self.config.experiment_id}")
        
        from sklearn.metrics import (
            precision_score, recall_score, f1_score, roc_auc_score,
            confusion_matrix
        )
        
        # Get predictions from both variants
        pred_a = variant_a_model.predict(variant_a_data)
        pred_a_proba = variant_a_model.predict_proba(variant_a_data)[:, 1]
        
        pred_b = variant_b_model.predict(variant_b_data)
        pred_b_proba = variant_b_model.predict_proba(variant_b_data)[:, 1]
        
        # Calculate metrics
        metrics_a = self._calculate_metrics(y_true, pred_a, pred_a_proba)
        metrics_b = self._calculate_metrics(y_true, pred_b, pred_b_proba)
        
        # Statistical test based on metric type
        if self.config.metric in ['precision', 'recall']:
            # Use proportion test
            tp_a = np.sum((pred_a == 1) & (y_true == 1))
            total_pos_a = np.sum(pred_a == 1) if self.config.metric == 'precision' else np.sum(y_true == 1)
            
            tp_b = np.sum((pred_b == 1) & (y_true == 1))
            total_pos_b = np.sum(pred_b == 1) if self.config.metric == 'precision' else np.sum(y_true == 1)
            
            test_result = self.tester.two_proportion_ztest(
                tp_a, total_pos_a, tp_b, total_pos_b,
                alpha=self.config.alpha
            )
        else:
            # Use t-test for continuous metrics
            scores_a = np.tile(metrics_a[self.config.metric], len(y_true))
            scores_b = np.tile(metrics_b[self.config.metric], len(y_true))
            
            test_result = self.tester.two_sample_ttest(
                scores_a, scores_b,
                alpha=self.config.alpha
            )
        
        self.results = {
            "experiment_id": self.config.experiment_id,
            "created_at": self.created_at.isoformat(),
            "variant_a": {
                "name": self.config.variant_a_name,
                "metrics": metrics_a,
            },
            "variant_b": {
                "name": self.config.variant_b_name,
                "metrics": metrics_b,
            },
            "statistical_test": test_result,
            "sample_size_required": self.tester.calculate_sample_size(
                metrics_a[self.config.metric],
                self.config.expected_effect_size,
                self.config.alpha,
                self.config.power
            ),
            "recommendation": self._generate_recommendation(test_result, metrics_a, metrics_b),
        }
        
        logger.info(f"✅ A/B test complete")
        self._print_results()
        
        return self.results
    
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_proba: np.ndarray) -> Dict:
        """Calculate comprehensive metrics"""
        
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        return {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "accuracy": np.mean(y_pred == y_true),
        }
    
    
    def _generate_recommendation(self, test_result: Dict, metrics_a: Dict, metrics_b: Dict) -> str:
        """Generate deployment recommendation based on test results"""
        
        if not test_result["significant"]:
            return "No significant difference. Choose based on other criteria (latency, cost, etc.)"
        
        variant_b_better = test_result["effect_size"] > 0
        metric_name = self.config.metric
        
        if variant_b_better:
            improvement = test_result.get("effect_size_percent", test_result["effect_size"] * 100)
            return f"✅ Deploy {self.config.variant_b_name} ({metric_name} +{improvement:.2f}%)"
        else:
            improvement = -test_result.get("effect_size_percent", test_result["effect_size"] * 100)
            return f"⚠️ Keep {self.config.variant_a_name} ({metric_name} +{improvement:.2f}%)"
    
    
    def _print_results(self):
        """Pretty print results"""
        
        logger.info("\n" + "="*60)
        logger.info(f"📊 A/B TEST RESULTS: {self.config.experiment_id}")
        logger.info("="*60)
        
        logger.info(f"\n{self.config.variant_a_name}:")
        for metric, value in self.results["variant_a"]["metrics"].items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info(f"\n{self.config.variant_b_name}:")
        for metric, value in self.results["variant_b"]["metrics"].items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info(f"\n📈 Statistical Test ({self.results['statistical_test']['test_type']}):")
        logger.info(f"  p-value: {self.results['statistical_test']['p_value']:.4f}")
        logger.info(f"  Significant: {self.results['statistical_test']['significant']}")
        logger.info(f"  Effect size: {self.results['statistical_test'].get('effect_size', 'N/A')}")
        
        logger.info(f"\n🎯 Recommendation: {self.results['recommendation']}")
        logger.info("="*60 + "\n")


class ExperimentTracker:
    """
    Track and log all experiments for audit trail
    """
    
    def __init__(self, log_file: str = "experiments.jsonl"):
        self.log_file = log_file
    
    
    def log_experiment(self, results: Dict):
        """Log experiment results"""
        
        import json
        
        record = {
            **results,
            "logged_at": datetime.utcnow().isoformat()
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record) + "\n")
        
        logger.info(f"📝 Experiment logged to {self.log_file}")
    
    
    def get_experiment_history(self) -> pd.DataFrame:
        """Load experiment history"""
        
        import json
        
        records = []
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    records.append(json.loads(line))
        except FileNotFoundError:
            logger.warning(f"Experiment log file not found: {self.log_file}")
            return pd.DataFrame()
        
        return pd.DataFrame(records)

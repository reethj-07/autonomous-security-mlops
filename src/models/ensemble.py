"""
Advanced ensemble models combining multiple architectures.
Includes XGBoost, LightGBM, stacking, and blending approaches.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import StackingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostSecurityModel:
    """
    XGBoost classifier optimized for security anomaly detection
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 7,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 1.0,
        random_state: int = 42,
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            tree_method='hist',
            device='cpu',
            eval_metric='logloss',
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, early_stopping_rounds=50):
        """Train XGBoost model with early stopping"""
        
        self.model.fit(
            X, y,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        
        return self
    
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binary predictions"""
        return self.model.predict(X)
    
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability predictions"""
        return self.model.predict_proba(X)
    
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Extract feature importance scores"""
        
        importances = self.model.feature_importances_
        feature_names = self.model.get_booster().feature_names
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    
    def log_to_mlflow(self, model_name: str = "xgboost_security"):
        """Log model parameters and feature importance to MLflow"""
        
        params = self.model.get_params()
        mlflow.log_params(params)
        
        importance_df = self.get_feature_importance()
        for idx, row in importance_df.head(10).iterrows():
            mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])
        
        logger.info(f"✅ XGBoost model logged to MLflow")


class LightGBMSecurityModel:
    """
    LightGBM classifier optimized for security anomaly detection
    Fast training, efficient memory usage
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 7,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 1.0,
        random_state: int = 42,
    ):
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            verbose=-1,
            force_row_wise=True,
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, early_stopping_rounds=50):
        """Train LightGBM model"""
        
        self.model.fit(
            X, y,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        
        return self
    
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binary predictions"""
        return self.model.predict(X)
    
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability predictions"""
        return self.model.predict_proba(X)
    
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Extract feature importance scores"""
        
        importances = self.model.feature_importances_
        feature_names = self.model.feature_name_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    
    def log_to_mlflow(self, model_name: str = "lightgbm_security"):
        """Log model to MLflow"""
        
        params = self.model.get_params()
        mlflow.log_params(params)
        
        importance_df = self.get_feature_importance()
        for idx, row in importance_df.head(10).iterrows():
            mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])
        
        logger.info(f"✅ LightGBM model logged to MLflow")


class StackingEnsembleModel:
    """
    Stacking ensemble combining multiple base learners
    with a meta-learner (Logistic Regression)
    """
    
    def __init__(self, random_state: int = 42):
        
        # Base learners
        base_learners = [
            ('xgb', XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                random_state=random_state, eval_metric='logloss'
            )),
            ('lgb', LGBMClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                random_state=random_state, verbose=-1
            )),
            ('cat', CatBoostClassifier(
                iterations=100, depth=5, learning_rate=0.05,
                random_state=random_state, verbose=0
            )),
        ]
        
        # Meta-learner
        meta_learner = LogisticRegression(random_state=random_state, max_iter=1000)
        
        self.model = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )
    
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train stacking ensemble"""
        
        logger.info("🔄 Training stacking ensemble...")
        self.model.fit(X, y)
        logger.info("✅ Stacking ensemble trained")
        
        return self
    
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binary predictions"""
        return self.model.predict(X)
    
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability predictions"""
        return self.model.predict_proba(X)
    
    
    def get_base_learner_weights(self) -> dict:
        """Extract meta-learner coefficients (weights for base learners)"""
        
        meta_coefs = self.model.final_estimator_.coef_[0]
        base_names = [name for name, _ in self.model.estimators_]
        
        weights = dict(zip(base_names, meta_coefs))
        return weights
    
    
    def log_to_mlflow(self, model_name: str = "stacking_ensemble"):
        """Log ensemble to MLflow"""
        
        weights = self.get_base_learner_weights()
        mlflow.log_params({"model_type": "StackingEnsemble", "base_learners": 3})
        
        for learner, weight in weights.items():
            mlflow.log_metric(f"base_learner_weight_{learner}", float(weight))
        
        logger.info(f"✅ Stacking ensemble logged to MLflow")


class VotingEnsembleModel:
    """
    Simple voting ensemble combining multiple classifiers.
    Faster than stacking but slightly less effective.
    """
    
    def __init__(self, voting: str = 'soft', weights=None, random_state: int = 42):
        """
        Parameters:
            voting: 'hard' or 'soft'
            weights: List of weights for each estimator
        """
        
        estimators = [
            ('xgb', XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                random_state=random_state, eval_metric='logloss'
            )),
            ('lgb', LGBMClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                random_state=random_state, verbose=-1
            )),
            ('cat', CatBoostClassifier(
                iterations=100, depth=5, learning_rate=0.05,
                random_state=random_state, verbose=0
            )),
        ]
        
        self.model = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=-1
        )
    
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train voting ensemble"""
        
        logger.info("🔄 Training voting ensemble...")
        self.model.fit(X, y)
        logger.info("✅ Voting ensemble trained")
        
        return self
    
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binary predictions"""
        return self.model.predict(X)
    
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability predictions"""
        return self.model.predict_proba(X)
    
    
    def log_to_mlflow(self, model_name: str = "voting_ensemble"):
        """Log ensemble to MLflow"""
        
        mlflow.log_params({
            "model_type": "VotingEnsemble",
            "voting": self.model.voting,
            "base_learners": 3
        })
        
        logger.info(f"✅ Voting ensemble logged to MLflow")


class HybridEnsembleSelector:
    """
    Dynamically selects best ensemble model based on validation metrics
    """
    
    def __init__(self):
        self.models = {}
    
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        """
        Train all ensemble variants and return performance comparison
        """
        
        from sklearn.metrics import (
            precision_score, recall_score, f1_score, roc_auc_score,
            confusion_matrix
        )
        
        results = {}
        
        # XGBoost
        logger.info("🔄 Training XGBoost...")
        xgb_model = XGBoostSecurityModel()
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        xgb_pred = xgb_model.predict(X_val)
        xgb_proba = xgb_model.predict_proba(X_val)[:, 1]
        
        results['XGBoost'] = {
            'model': xgb_model,
            'precision': precision_score(y_val, xgb_pred),
            'recall': recall_score(y_val, xgb_pred),
            'f1': f1_score(y_val, xgb_pred),
            'roc_auc': roc_auc_score(y_val, xgb_proba),
        }
        
        # LightGBM
        logger.info("🔄 Training LightGBM...")
        lgb_model = LightGBMSecurityModel()
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        lgb_pred = lgb_model.predict(X_val)
        lgb_proba = lgb_model.predict_proba(X_val)[:, 1]
        
        results['LightGBM'] = {
            'model': lgb_model,
            'precision': precision_score(y_val, lgb_pred),
            'recall': recall_score(y_val, lgb_pred),
            'f1': f1_score(y_val, lgb_pred),
            'roc_auc': roc_auc_score(y_val, lgb_proba),
        }
        
        # Stacking Ensemble
        logger.info("🔄 Training Stacking Ensemble...")
        stacking_model = StackingEnsembleModel()
        stacking_model.fit(X_train, y_train)
        stacking_pred = stacking_model.predict(X_val)
        stacking_proba = stacking_model.predict_proba(X_val)[:, 1]
        
        results['Stacking'] = {
            'model': stacking_model,
            'precision': precision_score(y_val, stacking_pred),
            'recall': recall_score(y_val, stacking_pred),
            'f1': f1_score(y_val, stacking_pred),
            'roc_auc': roc_auc_score(y_val, stacking_proba),
        }
        
        # Voting Ensemble
        logger.info("🔄 Training Voting Ensemble...")
        voting_model = VotingEnsembleModel()
        voting_model.fit(X_train, y_train)
        voting_pred = voting_model.predict(X_val)
        voting_proba = voting_model.predict_proba(X_val)[:, 1]
        
        results['Voting'] = {
            'model': voting_model,
            'precision': precision_score(y_val, voting_pred),
            'recall': recall_score(y_val, voting_pred),
            'f1': f1_score(y_val, voting_pred),
            'roc_auc': roc_auc_score(y_val, voting_proba),
        }
        
        # Log comparison to MLflow
        comparison_df = pd.DataFrame({
            model_name: metrics for model_name, metrics in results.items()
        }).T
        
        logger.info("\n🏆 Model Performance Comparison:")
        logger.info(comparison_df.to_string())
        
        # Log each model separately
        for model_name, metrics_dict in results.items():
            with mlflow.start_run(nested=True):
                mlflow.log_param("model_type", model_name)
                for metric_name, metric_value in metrics_dict.items():
                    if metric_name != 'model':
                        mlflow.log_metric(metric_name, metric_value)
        
        self.models = results
        return results
    
    
    def get_best_model(self, metric: str = 'f1'):
        """
        Returns the best model based on specified metric
        
        Parameters:
            metric: 'precision', 'recall', 'f1', or 'roc_auc'
        """
        
        if not self.models:
            raise ValueError("No models trained yet. Call train_all_models first.")
        
        scores = {name: metrics[metric] for name, metrics in self.models.items()}
        best_model_name = max(scores, key=scores.get)
        
        logger.info(f"🏆 Best model: {best_model_name} ({metric}={scores[best_model_name]:.4f})")
        
        return best_model_name, self.models[best_model_name]['model']

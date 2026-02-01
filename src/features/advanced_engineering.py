"""
Advanced feature engineering with temporal patterns, LSTM-ready sequences,
and automated feature selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering combining domain expertise and automation
    """
    
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = []
    
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal patterns from time-series security logs
        """
        
        logger.info("📅 Extracting temporal features...")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_business_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)).astype(int)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Time decay features (recent activity more relevant)
        max_time = df['timestamp'].max()
        df['time_since_last_event_min'] = (max_time - df['timestamp']).dt.total_seconds() / 60
        df['time_decay_factor'] = np.exp(-df['time_since_last_event_min'] / 1440)  # 1 day half-life
        
        return df
    
    
    def create_sequence_features(self, df: pd.DataFrame, user_col: str = 'user_id',
                                seq_length: int = 5) -> pd.DataFrame:
        """
        Create sequence-based features capturing request patterns
        """
        
        logger.info("🔄 Creating sequence features...")
        
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        # Method transition patterns
        df['method_transitions'] = df.groupby(user_col)['method'].shift().ne(df['method']).astype(int)
        
        # Path change frequency
        df['path_changes'] = df.groupby(user_col)['path'].shift().ne(df['path']).astype(int)
        
        # Status code sequences (detect attack patterns)
        df['status_code_sequence'] = df.groupby(user_col)['status_code'].apply(
            lambda x: x.astype(str).str.cat(sep='')
        ).str.len()
        
        # Request count in sliding window (5 requests)
        df['request_count_window'] = df.groupby(user_col).cumcount() + 1
        df['request_count_window'] = df['request_count_window'].clip(upper=seq_length)
        
        # Same path repeated
        df['repeated_path_count'] = df.groupby([user_col, 'path']).cumcount()
        
        # Failed request streaks
        df['failed_requests_streak'] = (
            df.groupby(user_col)[(df['status_code'] >= 400)].cumcount().fillna(0)
        )
        
        return df
    
    
    def create_behavioral_features(self, df: pd.DataFrame, user_col: str = 'user_id') -> pd.DataFrame:
        """
        Extract behavioral anomaly indicators
        """
        
        logger.info("🧠 Creating behavioral features...")
        
        # User deviation from norm
        user_stats = df.groupby(user_col).agg({
            'latency_ms': ['mean', 'std'],
            'request': 'count',
            'status_code': lambda x: (x >= 400).sum() / len(x),  # error rate
        }).reset_index()
        
        user_stats.columns = ['user_id', 'avg_latency', 'std_latency', 'request_count', 'error_rate']
        
        df = df.merge(user_stats, on=user_col, how='left')
        
        # Latency deviation from user average
        df['latency_deviation'] = (
            (df['latency_ms'] - df['avg_latency']) / (df['std_latency'] + 1)
        ).clip(-5, 5)  # Cap at 5 std devs
        
        # Unusual request volume
        df['request_volume_z_score'] = (
            (df['request_count'] - df['request_count'].mean()) / (df['request_count'].std() + 1)
        ).clip(-5, 5)
        
        # Error rate deviation
        global_error_rate = (df['status_code'] >= 400).sum() / len(df)
        df['error_rate_deviation'] = (df['error_rate'] - global_error_rate).abs()
        
        return df
    
    
    def create_attack_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Domain-specific features for common attack patterns
        """
        
        logger.info("🛡️ Creating attack pattern features...")
        
        # SQL injection patterns
        sql_keywords = ['select', 'union', 'drop', 'insert', 'delete', 'update',
                       'or 1=1', 'exec', 'execute', 'script', '--', '/*', '*/']
        df['sql_injection_risk'] = df['request'].str.lower().str.contains(
            '|'.join(sql_keywords), regex=True
        ).astype(int)
        
        # XSS pattern detection
        xss_keywords = ['<script', 'onerror', 'onclick', 'javascript:', 'iframe', 'alert(']
        df['xss_risk'] = df['request'].str.lower().str.contains(
            '|'.join(xss_keywords), regex=True
        ).astype(int)
        
        # Path traversal patterns
        df['path_traversal_risk'] = df['path'].str.contains(
            r'(\.\./|\.\.\\|%2e%2e)', regex=True
        ).astype(int)
        
        # Brute force patterns (multiple failed attempts)
        df['brute_force_indicator'] = (df['status_code'] == 401).astype(int)
        
        # Admin path access
        admin_patterns = ['admin', 'wp-admin', 'wp-login', 'phpmyadmin', 'config']
        df['admin_access'] = df['path'].str.lower().str.contains(
            '|'.join(admin_patterns), regex=True
        ).astype(int)
        
        # Suspicious HTTP methods on sensitive paths
        dangerous_combinations = [
            (df['method'] == 'DELETE') & (df['admin_access'] == 1),
            (df['method'] == 'POST') & (df['path_traversal_risk'] == 1),
        ]
        df['dangerous_method_path'] = sum(dangerous_combinations).astype(int)
        
        return df
    
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Statistical aggregation features
        """
        
        logger.info("📊 Creating statistical features...")
        
        # Request length statistics
        df['request_length_percentile'] = (
            df['request'].str.len().rank(pct=True)
        )
        
        # Latency statistics
        df['latency_percentile'] = df['latency_ms'].rank(pct=True)
        df['high_latency_flag'] = (df['latency_ms'] > df['latency_ms'].quantile(0.95)).astype(int)
        
        # Entropy of requests (diversity indicator)
        def calculate_entropy(series):
            values = series.unique()
            if len(values) <= 1:
                return 0.0
            counts = series.value_counts()
            probabilities = counts / len(series)
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        df['request_entropy'] = df.groupby('user_id')['request'].transform(
            lambda x: calculate_entropy(x)
        )
        df['path_entropy'] = df.groupby('user_id')['path'].transform(
            lambda x: calculate_entropy(x)
        )
        
        return df
    
    
    def create_embedding_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create numerical embeddings of categorical features
        (hash-based encoding to capture semantic similarity)
        """
        
        logger.info("🔤 Creating embedding features...")
        
        # Hash-based embeddings for paths (maps similar paths close together)
        df['path_embedding_1'] = df['path'].apply(lambda x: hash(x) % 100)
        df['path_embedding_2'] = df['path'].apply(lambda x: hash(x + 'salt1') % 100)
        df['path_embedding_3'] = df['path'].apply(lambda x: hash(x + 'salt2') % 100)
        
        # Request pattern embeddings
        df['request_embedding_1'] = df['request'].apply(lambda x: hash(x[:50]) % 100)
        df['request_embedding_2'] = df['request'].apply(lambda x: hash(x[-50:]) % 100)
        
        # IP embedding (simplified)
        df['ip_embedding'] = df['ip_address'].apply(lambda x: hash(x) % 100)
        
        return df
    
    
    def select_best_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 20,
                           method: str = 'f_classif') -> pd.DataFrame:
        """
        Automated feature selection using statistical tests
        
        Parameters:
            X: Feature matrix
            y: Target labels
            n_features: Number of features to select
            method: 'f_classif' or 'mutual_info'
        """
        
        logger.info(f"🎯 Selecting top {n_features} features using {method}...")
        
        if method == 'f_classif':
            scoring_func = f_classif
        elif method == 'mutual_info':
            scoring_func = mutual_info_classif
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        self.feature_selector = SelectKBest(scoring_func, k=n_features)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = X.columns[selected_mask].tolist()
        
        logger.info(f"✅ Selected features: {self.selected_features}")
        
        # Log feature scores
        scores = self.feature_selector.scores_
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': scores
        }).sort_values('score', ascending=False)
        
        logger.info("\n📊 Top 10 features by score:")
        logger.info(feature_scores.head(10).to_string(index=False))
        
        return pd.DataFrame(X_selected, columns=self.selected_features)
    
    
    def prepare_lstm_sequences(self, df: pd.DataFrame, seq_length: int = 5,
                             user_col: str = 'user_id') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM models
        
        Returns:
            (sequences, labels) ready for neural networks
        """
        
        logger.info(f"🔢 Preparing LSTM sequences of length {seq_length}...")
        
        sequences = []
        labels = []
        
        for user_id in df[user_col].unique():
            user_data = df[df[user_col] == user_id].sort_values('timestamp')
            
            feature_cols = [col for col in df.columns if col not in [user_col, 'timestamp', 'label']]
            X = user_data[feature_cols].values
            y = user_data['label'].values if 'label' in user_data.columns else None
            
            # Create sliding windows
            for i in range(len(X) - seq_length + 1):
                sequences.append(X[i:i+seq_length])
                if y is not None:
                    labels.append(y[i+seq_length-1])  # Use last label as sequence label
        
        sequences = np.array(sequences)
        labels = np.array(labels) if labels else None
        
        logger.info(f"✅ Created {len(sequences)} sequences of shape {sequences[0].shape}")
        
        return sequences, labels
    
    
    def generate_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Master method combining all feature engineering steps
        """
        
        logger.info("🚀 Starting comprehensive feature engineering pipeline...")
        
        # Apply all feature engineering steps
        df = self.create_temporal_features(df)
        df = self.create_sequence_features(df)
        df = self.create_behavioral_features(df)
        df = self.create_attack_pattern_features(df)
        df = self.create_statistical_features(df)
        df = self.create_embedding_features(df)
        
        logger.info(f"✅ Feature engineering complete. Total features: {len(df.columns)}")
        logger.info(f"📋 Features: {list(df.columns)}")
        
        return df

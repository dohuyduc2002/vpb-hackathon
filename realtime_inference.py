import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

os.environ['DGLBACKEND'] = 'pytorch'

import torch as th
import dgl
import numpy as np
import pandas as pd
import pickle
import joblib
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from gnn.pytorch_model import HeteroRGCN
from train_hybrid_model import HybridFraudDetector


class RealTimeFraudDetector:
    """
    Real-time fraud detection system using the trained hybrid model
    """
    
    def __init__(self, model_dir, device='cpu'):
        """
        Initialize the real-time fraud detector
        
        Args:
            model_dir: Directory containing trained model files
            device: Device for inference (cpu/cuda)
        """
        self.model_dir = model_dir
        self.device = device
        self.hybrid_model = None
        self.metadata = None
        self.is_loaded = False
        
        # Performance monitoring
        self.inference_times = []
        self.prediction_count = 0
        
    def load_model(self):
        print("Loading hybrid fraud detection model...")
        
        try:
            # Initialize and load hybrid model
            self.hybrid_model = HybridFraudDetector(
                ntype_dict={},  
                etypes=[],     
                gnn_config={},  
                device=self.device
            )
            
            self.hybrid_model.load_model(self.model_dir)
            
            # Load training metadata
            metadata_path = os.path.join(self.model_dir, 'training_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            else:
                print("Warning: Training metadata not found. Some features may be limited.")
                self.metadata = {}
            
            self.is_loaded = True
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_transaction(self, transaction_data, graph_context=None):
        """
        Preprocess a single transaction for inference
        
        Args:
            transaction_data: Dict containing transaction features
            graph_context: Optional graph context (for incremental updates)
            
        Returns:
            Preprocessed features ready for model input
        """
        # Extract numerical features
        feature_columns = [
            'amount', 'hour', 'day_of_week', 'days_since_last_transaction',
            'transaction_count_1h', 'transaction_count_24h', 'avg_amount_1h',
            'avg_amount_24h', 'merchant_risk_score', 'user_risk_score'
        ]
        
        features = []
        for col in feature_columns:
            features.append(transaction_data.get(col, 0.0))
        
        features = th.tensor([features], dtype=th.float32)

        if self.metadata and 'feature_mean' in self.metadata:
            mean = self.metadata['feature_mean']
            std = self.metadata['feature_std']
            features = (features - mean) / std
        
        return features
    
    def create_inference_graph(self, transaction_data, historical_graph=None):
        """
        Create or update graph for inference
        
        Args:
            transaction_data: Current transaction data
            historical_graph: Existing graph to update (optional)
            
        Returns:
            DGL graph for inference
        """       
        if historical_graph is not None:
            # Update existing graph with new transaction
            return self._update_graph_incremental(historical_graph, transaction_data)
        else:
            # Create minimal graph for this transaction
            return self._create_minimal_graph(transaction_data)
    
    def _create_minimal_graph(self, transaction_data):
        """Create a minimal graph for a single transaction"""
        
        # Create a simple graph with user, merchant, and transaction nodes
        graph_data = {
            ('user', 'transacts', 'merchant'): ([0], [0]),
            ('merchant', 'receives', 'user'): ([0], [0]),
            ('user', 'makes', 'transaction'): ([0], [0]),
            ('transaction', 'belongs_to', 'user'): ([0], [0])
        }
        
        g = dgl.heterograph(graph_data)
        
        # Add dummy features for non-target nodes
        embedding_dim = self.hybrid_model.gnn_config.get('in_feats', 64)
        
        g.nodes['user'].data['features'] = th.zeros(1, embedding_dim)
        g.nodes['merchant'].data['features'] = th.zeros(1, embedding_dim)
        
        return g
    
    def _update_graph_incremental(self, graph, transaction_data):
        """Update graph incrementally with new transaction"""
        return graph
    
    def predict_single_transaction(self, transaction_data, return_probabilities=True):
        """
        Predict fraud probability for a single transaction
        
        Args:
            transaction_data: Dict containing transaction features
            return_probabilities: Whether to return detailed probabilities
            
        Returns:
            Fraud prediction result
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Preprocess transaction
            features = self.preprocess_transaction(transaction_data)
            
            # Create inference graph
            g = self.create_inference_graph(transaction_data)
            
            # Make prediction
            result = self.hybrid_model.predict_hybrid(g, features)
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.prediction_count += 1
            
            prediction_result = {
                'transaction_id': transaction_data.get('transaction_id', 'unknown'),
                'is_fraud': bool(result['predictions'][0]),
                'fraud_probability': float(result['probabilities'][0]),
                'inference_time_ms': inference_time * 1000,
                'timestamp': datetime.now().isoformat()
            }
            
            if return_probabilities:
                prediction_result.update({
                    'gnn_probability': float(result['gnn_proba'][0]),
                    'xgb_probability': float(result['xgb_proba'][0]),
                    'risk_level': self._get_risk_level(result['probabilities'][0])
                })
            
            return prediction_result
            
        except Exception as e:
            return {
                'transaction_id': transaction_data.get('transaction_id', 'unknown'),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch_transactions(self, transactions_list, batch_size=100):
        """
        Predict fraud for a batch of transactions
        
        Args:
            transactions_list: List of transaction dictionaries
            batch_size: Number of transactions to process at once
            
        Returns:
            List of prediction results
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = []
        
        for i in range(0, len(transactions_list), batch_size):
            batch = transactions_list[i:i + batch_size]
            batch_results = []
            
            for transaction in batch:
                result = self.predict_single_transaction(transaction)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            if len(results) % 1000 == 0:
                print(f"Processed {len(results)} transactions...")
        
        return results
    
    def _get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability < 0.3:
            return 'LOW'
        elif probability < 0.7:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.inference_times:
            return {"message": "No predictions made yet"}
        
        times_ms = [t * 1000 for t in self.inference_times]
        
        return {
            'total_predictions': self.prediction_count,
            'avg_inference_time_ms': np.mean(times_ms),
            'median_inference_time_ms': np.median(times_ms),
            'min_inference_time_ms': np.min(times_ms),
            'max_inference_time_ms': np.max(times_ms),
            'p95_inference_time_ms': np.percentile(times_ms, 95),
            'p99_inference_time_ms': np.percentile(times_ms, 99)
        }
    
    def explain_prediction(self, transaction_data):
        """
        Provide explanation for fraud prediction
        
        Args:
            transaction_data: Transaction to explain
            
        Returns:
            Explanation dictionary
        """
        result = self.predict_single_transaction(transaction_data, return_probabilities=True)
        
        if 'error' in result:
            return result
        
        feature_importance = None
        if hasattr(self.hybrid_model.xgb_model, 'feature_importances_'):
            feature_importance = self.hybrid_model.xgb_model.feature_importances_
        
        explanation = {
            'transaction_id': result['transaction_id'],
            'prediction': result['is_fraud'],
            'confidence': result['fraud_probability'],
            'risk_factors': self._identify_risk_factors(transaction_data, result),
            'model_contributions': {
                'gnn_score': result['gnn_probability'],
                'xgb_score': result['xgb_probability'],
                'ensemble_weights': self.hybrid_model.ensemble_weights
            }
        }
        
        if feature_importance is not None:
            explanation['feature_importance'] = feature_importance.tolist()
        
        return explanation
    
    def _identify_risk_factors(self, transaction_data, prediction_result):
        """Identify key risk factors for this transaction"""
        risk_factors = []
        
        # High amount
        amount = transaction_data.get('amount', 0)
        if amount > 1000:
            risk_factors.append(f"High transaction amount: ${amount:.2f}")
        
        # Unusual time
        hour = transaction_data.get('hour', 12)
        if hour < 6 or hour > 22:
            risk_factors.append(f"Unusual transaction time: {hour:02d}:00")
        
        # High frequency
        count_1h = transaction_data.get('transaction_count_1h', 0)
        if count_1h > 5:
            risk_factors.append(f"High transaction frequency: {count_1h} in last hour")
        
        # Risk scores
        merchant_risk = transaction_data.get('merchant_risk_score', 0)
        if merchant_risk > 0.7:
            risk_factors.append(f"High-risk merchant: {merchant_risk:.2f}")
        
        user_risk = transaction_data.get('user_risk_score', 0)
        if user_risk > 0.7:
            risk_factors.append(f"High-risk user: {user_risk:.2f}")
        
        return risk_factors


class FraudDetectionAPI:
    """
    Simple API wrapper for the fraud detection system
    """
    
    def __init__(self, model_dir, device='cpu'):
        self.detector = RealTimeFraudDetector(model_dir, device)
        self.detector.load_model()
        
    def health_check(self):
        """API health check"""
        return {
            'status': 'healthy' if self.detector.is_loaded else 'unhealthy',
            'model_loaded': self.detector.is_loaded,
            'device': str(self.detector.device),
            'timestamp': datetime.now().isoformat()
        }
    
    def predict(self, transaction_data):
        """API endpoint for fraud prediction"""
        return self.detector.predict_single_transaction(transaction_data)
    
    def explain(self, transaction_data):
        """API endpoint for prediction explanation"""
        return self.detector.explain_prediction(transaction_data)
    
    def stats(self):
        """API endpoint for performance statistics"""
        return self.detector.get_performance_stats()


def create_sample_transaction():
    """Create a sample transaction for testing"""
    return {
        'transaction_id': 'TXN_12345',
        'user_id': 'user_789',
        'merchant_id': 'merchant_456',
        'amount': 150.75,
        'hour': 14,
        'day_of_week': 3,
        'days_since_last_transaction': 2,
        'transaction_count_1h': 1,
        'transaction_count_24h': 3,
        'avg_amount_1h': 150.75,
        'avg_amount_24h': 95.50,
        'merchant_risk_score': 0.3,
        'user_risk_score': 0.2
    }


def main():
    """
    Demo script for real-time fraud detection
    """
    print("Real-Time Fraud Detection Demo")
    print("=" * 40)
    
    # Initialize detector
    model_dir = './output'  # Adjust path as needed
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    
    try:
        detector = RealTimeFraudDetector(model_dir, device)
        detector.load_model()
        
        print(f"Model loaded successfully on {device}")
        
        # Create sample transaction
        sample_transaction = create_sample_transaction()
        print(f"\nSample transaction: {sample_transaction}")
        
        # Single prediction
        print("\n--- Single Prediction ---")
        result = detector.predict_single_transaction(sample_transaction)
        print(f"Prediction result: {result}")
        
        # Explanation
        print("\n--- Prediction Explanation ---")
        explanation = detector.explain_prediction(sample_transaction)
        print(f"Explanation: {explanation}")
        
        # Performance stats
        print("\n--- Performance Statistics ---")
        stats = detector.get_performance_stats()
        print(f"Performance: {stats}")
        
        # Batch prediction demo
        print("\n--- Batch Prediction Demo ---")
        batch_transactions = [create_sample_transaction() for _ in range(10)]
        batch_results = detector.predict_batch_transactions(batch_transactions)
        print(f"Processed {len(batch_results)} transactions")
        
        # Updated performance stats
        print("\n--- Updated Performance Statistics ---")
        updated_stats = detector.get_performance_stats()
        print(f"Updated performance: {updated_stats}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure the model has been trained and saved first using train_hybrid_model.py")


if __name__ == '__main__':
    main() 
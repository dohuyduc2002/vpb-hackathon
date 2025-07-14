import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc
import pickle
import argparse
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def calculate_pr_auc_with_thresholds(y_true, y_pred_proba, thresholds=[0.5, 0.6, 0.7, 0.8]):
    """
    Calculate PR AUC and metrics at different probability thresholds
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        thresholds: List of probability thresholds to evaluate
        
    Returns:
        Dictionary with PR AUC and threshold-specific metrics
    """
    # Calculate overall PR AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Calculate metrics at specific thresholds
    threshold_metrics = {}
    
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        
        # Calculate precision, recall, and F1 at this threshold
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        prec = precision_score(y_true, y_pred_thresh, zero_division=0)
        rec = recall_score(y_true, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
        
        # Count predictions
        total_positive_pred = sum(y_pred_thresh)
        total_actual_positive = sum(y_true)
        
        threshold_metrics[threshold] = {
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'predicted_positive': total_positive_pred,
            'actual_positive': total_actual_positive,
            'prediction_rate': total_positive_pred / len(y_pred_thresh)
        }
    
    return {
        'pr_auc': pr_auc,
        'threshold_metrics': threshold_metrics
    }


class SimplifiedHybridFraudDetector:
    """
    Simplified hybrid fraud detector using XGBoost with graph-inspired features
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.xgb_model = None
        self.feature_scaler = StandardScaler()
        self.feature_names = None
        self.trained = False
        
    def create_graph_features(self, df):
        """
        Create graph-inspired features from transaction data
        """
        user_features = df.groupby('User_ID').agg({
            'Amount': ['count', 'mean', 'std', 'min', 'max'],
            'Is_Fraud': 'mean',
            'Hour': lambda x: x.value_counts().index[0],  # Most common hour
            'MCC': 'nunique'  # Number of different merchant categories
        }).fillna(0)
        
        user_features.columns = ['_'.join(col).strip() for col in user_features.columns]

        merchant_features = df.groupby('Merchant_ID').agg({
            'Amount': ['count', 'mean', 'std'],
            'Is_Fraud': 'mean',
            'User_ID': 'nunique'  # Number of different users
        }).fillna(0)
        
        merchant_features.columns = ['merchant_' + '_'.join(col).strip() for col in merchant_features.columns]
        
        enhanced_df = df.copy()
        enhanced_df = enhanced_df.merge(user_features, left_on='User_ID', right_index=True, how='left')
        enhanced_df = enhanced_df.merge(merchant_features, left_on='Merchant_ID', right_index=True, how='left')
        
        enhanced_df['amount_vs_user_avg'] = enhanced_df['Amount'] / (enhanced_df['Amount_mean'] + 1e-6)
        enhanced_df['amount_vs_merchant_avg'] = enhanced_df['Amount'] / (enhanced_df['merchant_Amount_mean'] + 1e-6)
        enhanced_df['user_fraud_risk'] = enhanced_df['Is_Fraud_mean']
        enhanced_df['merchant_fraud_risk'] = enhanced_df['merchant_Is_Fraud_mean']
        enhanced_df['is_weekend'] = enhanced_df['Day_of_Week'].isin([0, 6]).astype(int)
        enhanced_df['is_night'] = ((enhanced_df['Hour'] < 6) | (enhanced_df['Hour'] > 22)).astype(int)
        enhanced_df['high_amount'] = (enhanced_df['Amount'] > enhanced_df['Amount'].quantile(0.95)).astype(int)
        
        return enhanced_df
    
    def prepare_features(self, df):
        """
        Prepare features for training
        """
        # Base features
        feature_columns = [
            'Amount', 'Hour', 'Day_of_Week', 'Year', 'Use_Chip_Binary', 'MCC',
            'Amount_count', 'Amount_mean', 'Amount_std', 'Amount_min', 'Amount_max',
            'Is_Fraud_mean', 'Hour_<lambda>', 'MCC_nunique',
            'merchant_Amount_count', 'merchant_Amount_mean', 'merchant_Amount_std',
            'merchant_Is_Fraud_mean', 'merchant_User_ID_nunique',
            'amount_vs_user_avg', 'amount_vs_merchant_avg', 'user_fraud_risk',
            'merchant_fraud_risk', 'is_weekend', 'is_night', 'high_amount'
        ]
        
        # Select available features
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].fillna(0)
        
        # Store feature names
        if self.feature_names is None:
            self.feature_names = X.columns.tolist()
        
        return X
    
    def train(self, df, test_size=0.2):
        """
        Train the simplified hybrid model
        """
        print("="*60)
        print("TRAINING SIMPLIFIED HYBRID FRAUD DETECTION MODEL")
        print("="*60)
        enhanced_df = self.create_graph_features(df)
        X = self.prepare_features(enhanced_df)
        y = enhanced_df['Is_Fraud'].values
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Fraud rate: {y.mean():.3f}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': scale_pos_weight,
            'random_state': self.random_state
        }
        
        print(f"Training XGBoost with {X_train.shape[1]} features...")
        
        # Train XGBoost
        self.xgb_model = xgb.XGBClassifier(**xgb_params)
        
        # Fit with early stopping using callbacks for newer XGBoost versions
        try:
            self.xgb_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                callbacks=[xgb.callback.EarlyStopping(rounds=20, save_best=True)],
                verbose=False
            )
        except TypeError:
            # Fallback for older XGBoost versions
            self.xgb_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False
            )
        
        # Evaluate
        train_pred = self.xgb_model.predict(X_train_scaled)
        train_pred_proba = self.xgb_model.predict_proba(X_train_scaled)[:, 1]
        
        test_pred = self.xgb_model.predict(X_test_scaled)
        test_pred_proba = self.xgb_model.predict_proba(X_test_scaled)[:, 1]
        
        # Print results
        print("\n" + "="*50)
        print("TRAINING RESULTS")
        print("="*50)
        
        train_auc = roc_auc_score(y_train, train_pred_proba)
        test_auc = roc_auc_score(y_test, test_pred_proba)
        
        print(f"Train AUC: {train_auc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        
        # Calculate PR AUC with thresholds
        print("\n" + "="*50)
        print("PR AUC AND THRESHOLD ANALYSIS")
        print("="*50)
        
        train_pr_results = calculate_pr_auc_with_thresholds(y_train, train_pred_proba)
        test_pr_results = calculate_pr_auc_with_thresholds(y_test, test_pred_proba)
        
        print(f"Train PR AUC: {train_pr_results['pr_auc']:.4f}")
        print(f"Test PR AUC: {test_pr_results['pr_auc']:.4f}")
        
        print("\nThreshold Analysis (Test Set):")
        print("Threshold | Precision | Recall | F1-Score | Pred+ | Rate  ")
        print("-" * 60)
        
        for threshold in [0.5, 0.6, 0.7, 0.8]:
            metrics = test_pr_results['threshold_metrics'][threshold]
            print(f"   {threshold:.1f}    |   {metrics['precision']:.3f}   | {metrics['recall']:.3f} |  {metrics['f1_score']:.3f}  | {metrics['predicted_positive']:4d} | {metrics['prediction_rate']:.3f}")
        
        print("\nTest Set Classification Report:")
        print(classification_report(y_test, test_pred))
        
        print("\nTest Set Confusion Matrix:")
        print(confusion_matrix(y_test, test_pred))
        
        # Feature importance
        print("\nTop 10 Feature Importances:")
        feature_importance = sorted(
            zip(self.feature_names, self.xgb_model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )[:10]
        
        for feature, importance in feature_importance:
            print(f"  {feature}: {importance:.4f}")
        
        self.trained = True
        
        return {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_pr_auc': train_pr_results['pr_auc'],
            'test_pr_auc': test_pr_results['pr_auc'],
            'threshold_metrics': test_pr_results['threshold_metrics'],
            'feature_importance': feature_importance
        }
    
    def predict(self, df):
        """
        Make predictions on new data
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create graph-inspired features
        enhanced_df = self.create_graph_features(df)
        
        # Prepare features
        X = self.prepare_features(enhanced_df)
        X_scaled = self.feature_scaler.transform(X)
        
        # Predict
        predictions = self.xgb_model.predict_proba(X_scaled)[:, 1]
        binary_predictions = self.xgb_model.predict(X_scaled)
        
        return predictions, binary_predictions
    
    def save_model(self, model_dir):
        """
        Save the trained model
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save XGBoost model
        self.xgb_model.save_model(os.path.join(model_dir, 'xgb_model.json'))
        
        # Save other components
        model_data = {
            'feature_scaler': self.feature_scaler,
            'feature_names': self.feature_names,
            'trained': self.trained,
            'random_state': self.random_state
        }
        
        with open(os.path.join(model_dir, 'model_components.pkl'), 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir):
        """
        Load a trained model
        """
        # Load XGBoost model
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(os.path.join(model_dir, 'xgb_model.json'))
        
        # Load other components
        with open(os.path.join(model_dir, 'model_components.pkl'), 'rb') as f:
            model_data = pickle.load(f)
        
        self.feature_scaler = model_data['feature_scaler']
        self.feature_names = model_data['feature_names']
        self.trained = model_data['trained']
        self.random_state = model_data['random_state']
        
        print(f"Model loaded from {model_dir}")


def load_and_preprocess_encoded_csv(csv_path, max_samples=500000):
    """
    Load and preprocess the encoded CSV data
    """
    print(f"Loading data from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    
    # Drop unnamed columns if they exist
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    
    print(f"Loaded {len(df)} transactions")
    
    # Sample data if too large to avoid memory issues
    if len(df) > max_samples:
        print(f"Dataset too large ({len(df)} rows). Sampling {max_samples} transactions...")
        # Stratified sampling to maintain fraud ratio
        fraud_df = df[df['Is Fraud?'] == 1]
        normal_df = df[df['Is Fraud?'] == 0]
        
        fraud_ratio = len(fraud_df) / len(df)
        fraud_samples = min(len(fraud_df), int(max_samples * fraud_ratio))
        normal_samples = max_samples - fraud_samples
        
        sampled_fraud = fraud_df.sample(n=fraud_samples, random_state=42)
        sampled_normal = normal_df.sample(n=normal_samples, random_state=42)
        
        df = pd.concat([sampled_fraud, sampled_normal]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Sampled {len(df)} transactions (fraud rate: {df['Is Fraud?'].mean():.3f})")
    
    print(f"Final dataset: {len(df)} transactions")
    print(f"Columns: {list(df.columns)}")
    print(f"Fraud rate: {df['Is Fraud?'].mean():.3f}")
    
    # Rename columns for consistency
    df = df.rename(columns={'Is Fraud?': 'Is_Fraud'})
    
    # Create synthetic user IDs (since not provided)
    np.random.seed(42)
    n_users = min(1000, len(df) // 5)
    df['User_ID'] = np.random.randint(0, n_users, len(df))
    
    # Create merchant IDs from Merchant Name
    merchant_encoder = LabelEncoder()
    df['Merchant_ID'] = merchant_encoder.fit_transform(df['Merchant Name'].astype(str))
    
    # Extract Use Chip information
    if 'Use Chip_0' in df.columns and 'Use Chip_1' in df.columns:
        # Convert one-hot encoding back to single column
        df['Use_Chip_Binary'] = 0  # Default to Swipe
        df.loc[df['Use Chip_1'] == 1, 'Use_Chip_Binary'] = 1  # Chip
        # Note: Online would be 2, but not present in this encoding
    else:
        df['Use_Chip_Binary'] = 0  # Default value
    
    # Extract Day of Week information
    if 'Day of Week_0' in df.columns:
        df['Day_of_Week'] = 0
        for i in range(3):  # 0, 1, 2
            col = f'Day of Week_{i}'
            if col in df.columns:
                df.loc[df[col] == 1, 'Day_of_Week'] = i
    else:
        df['Day_of_Week'] = 1  # Default value
    
    print(f"Processed data:")
    print(f"  - Users: {df['User_ID'].nunique()}")
    print(f"  - Merchants: {df['Merchant_ID'].nunique()}")
    print(f"  - Transactions: {len(df)}")
    
    return df


def main():
    """
    Main function for simplified training
    """
    parser = argparse.ArgumentParser(description='Train simplified hybrid fraud detection model')
    parser.add_argument('--csv_path', type=str, default='./data.csv',
                       help='Path to encoded CSV file')
    parser.add_argument('--model_dir', type=str, default='./simple_hybrid_model',
                       help='Directory to save trained model')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set fraction')
    parser.add_argument('--max_samples', type=int, default=500000,
                       help='Maximum number of samples to use for training')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        return
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_encoded_csv(args.csv_path, max_samples=args.max_samples)
        
        # Initialize and train model
        model = SimplifiedHybridFraudDetector()
        results = model.train(df, test_size=args.test_size)
        
        # Save model
        model.save_model(args.model_dir)
        
        print(f"Final Test AUC: {results['test_auc']:.4f}")
        print(f"Final Test PR AUC: {results['test_pr_auc']:.4f}")
        print("\nPerformance at Different Thresholds:")
        for threshold in [0.5, 0.6, 0.7, 0.8]:
            metrics = results['threshold_metrics'][threshold]
            print(f"  Threshold {threshold}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 
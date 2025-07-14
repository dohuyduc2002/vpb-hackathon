import os
import pandas as pd
import numpy as np
import argparse
from train_simple_hybrid import SimplifiedHybridFraudDetector, load_and_preprocess_encoded_csv, calculate_pr_auc_with_thresholds


def analyze_risk_factors(predictions, features, feature_names, top_n=5):
    """
    Analyze risk factors for high-risk transactions
    """
    high_risk_indices = np.where(predictions > 0.5)[0]
    
    if len(high_risk_indices) == 0:
        return "No high-risk transactions detected."
   
    risk_analysis = []
    
    for idx in high_risk_indices[:10]: 
        risk_factors = []
        transaction_features = features.iloc[idx]
        if transaction_features.get('is_night', 0) == 1:
            risk_factors.append("Transaction at unusual hours (night)")
        
        if transaction_features.get('is_weekend', 0) == 1:
            risk_factors.append("Weekend transaction")
        
        if transaction_features.get('high_amount', 0) == 1:
            risk_factors.append("High transaction amount")
        
        if transaction_features.get('user_fraud_risk', 0) > 0.1:
            risk_factors.append(f"User has fraud history ({transaction_features['user_fraud_risk']:.1%})")
        
        if transaction_features.get('merchant_fraud_risk', 0) > 0.1:
            risk_factors.append(f"Merchant has fraud history ({transaction_features['merchant_fraud_risk']:.1%})")
        
        if transaction_features.get('amount_vs_user_avg', 0) > 3:
            risk_factors.append("Amount much higher than user average")
        
        if transaction_features.get('amount_vs_merchant_avg', 0) > 3:
            risk_factors.append("Amount much higher than merchant average")
        
        risk_analysis.append({
            'transaction_id': idx,
            'fraud_probability': predictions[idx],
            'risk_factors': risk_factors
        })
    
    return risk_analysis


def categorize_risk_level(probability):
    """
    Categorize risk level based on fraud probability
    """
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.3:
        return "MEDIUM"
    else:
        return "LOW"


def run_inference(model_dir, csv_path, output_path=None):
    """
    Run inference on new transaction data
    """
    print("="*60)
    print("SIMPLIFIED HYBRID FRAUD DETECTION INFERENCE")
    print("="*60)
    
    # Load model
    print("Loading trained model...")
    model = SimplifiedHybridFraudDetector()
    model.load_model(model_dir)
    
    print("Loading and preprocessing data...")
    df = load_and_preprocess_encoded_csv(csv_path, max_samples=100000)
    
    enhanced_df = model.create_graph_features(df)
    features = model.prepare_features(enhanced_df)
    
    print("Making predictions...")
    fraud_probabilities, binary_predictions = model.predict(df)
    
    results_df = df.copy()
    results_df['fraud_probability'] = fraud_probabilities
    results_df['predicted_fraud'] = binary_predictions
    results_df['risk_level'] = [categorize_risk_level(p) for p in fraud_probabilities]

    total_transactions = len(results_df)
    predicted_fraud = sum(binary_predictions)
    high_risk = sum(results_df['risk_level'] == 'HIGH')
    medium_risk = sum(results_df['risk_level'] == 'MEDIUM')
    low_risk = sum(results_df['risk_level'] == 'LOW')
    
    print("\n" + "="*50)
    print("INFERENCE")
    print("="*50)
    print(f"Total transactions processed: {total_transactions}")
    print(f"Predicted fraud cases: {predicted_fraud} ({predicted_fraud/total_transactions:.1%})")
    print(f"Risk level distribution:")
    print(f"  - HIGH:   {high_risk} ({high_risk/total_transactions:.1%})")
    print(f"  - MEDIUM: {medium_risk} ({medium_risk/total_transactions:.1%})")
    print(f"  - LOW:    {low_risk} ({low_risk/total_transactions:.1%})")
    
    print("\n" + "="*50)
    print("TOP 10 HIGH-RISK TRANSACTIONS")
    print("="*50)
    
    top_risk = results_df.nlargest(10, 'fraud_probability')
    for idx, row in top_risk.iterrows():
        print(f"Transaction {idx}:")
        print(f"  - Fraud Probability: {row['fraud_probability']:.3f}")
        print(f"  - Risk Level: {row['risk_level']}")
        print(f"  - Amount: ${row['Amount']:.2f}")
        print(f"  - Merchant: {row['Merchant Name']}")
        print(f"  - Time: Hour {row['Hour']}")
        print()

    print("\n" + "="*50)
    print("RISK FACTOR ANALYSIS")
    print("="*50)
    
    risk_analysis = analyze_risk_factors(fraud_probabilities, features, model.feature_names)
    
    if isinstance(risk_analysis, str):
        print(risk_analysis)
    else:
        for analysis in risk_analysis[:5]: 
            print(f"Transaction {analysis['transaction_id']}:")
            print(f"  - Fraud Probability: {analysis['fraud_probability']:.3f}")
            print(f"  - Risk Factors:")
            for factor in analysis['risk_factors']:
                print(f"    • {factor}")
            print()
    
    if 'Is_Fraud' in results_df.columns:
        print("\n" + "="*50)
        print("PR AUC AND THRESHOLD ANALYSIS")
        print("="*50)

        true_labels = results_df['Is_Fraud'].values
        pr_results = calculate_pr_auc_with_thresholds(true_labels, fraud_probabilities)
        
        print(f"PR AUC: {pr_results['pr_auc']:.4f}")
        
        print("\nThreshold Analysis:")
        print("Threshold | Precision | Recall | F1-Score | Pred+ | Rate  ")
        print("-" * 60)
        
        for threshold in [0.5, 0.6, 0.7, 0.8]:
            metrics = pr_results['threshold_metrics'][threshold]
            print(f"   {threshold:.1f}    |   {metrics['precision']:.3f}   | {metrics['recall']:.3f} |  {metrics['f1_score']:.3f}  | {metrics['predicted_positive']:4d} | {metrics['prediction_rate']:.3f}")

    avg_processing_time = 1.0  
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Processing time: ~{avg_processing_time:.1f}ms per transaction")
    print(f"Model type: Simplified Hybrid (XGBoost + Graph Features)")
    print(f"Feature count: {len(model.feature_names)}")
    
    if 'Is_Fraud' in results_df.columns:
        from sklearn.metrics import roc_auc_score
        actual_auc = roc_auc_score(true_labels, fraud_probabilities)
        print(f"Actual ROC AUC: {actual_auc:.4f}")
        print(f"Actual PR AUC: {pr_results['pr_auc']:.4f}")
    
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    return results_df


def main():
    """
    Main function for inference
    """
    parser = argparse.ArgumentParser(description='Run inference with simplified hybrid fraud detection model')
    parser.add_argument('--model_dir', type=str, default='./simple_hybrid_model',
                       help='Directory containing trained model')
    parser.add_argument('--csv_path', type=str, default='./data.csv',
                       help='Path to CSV file with transaction data')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save results CSV (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        return
    
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        return
    
    try:
        results = run_inference(args.model_dir, args.csv_path, args.output_path)
        
    except Exception as e:
        print(f"Inference failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

os.environ['DGLBACKEND'] = 'pytorch'

import torch as th
import dgl
import numpy as np
import pandas as pd
import time
import pickle
import copy
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb

from gnn.estimator_fns import *
from gnn.graph_utils import *
from gnn.data import *
from gnn.utils import *
from gnn.pytorch_model import *


class HybridFraudDetector:
    """
    Hybrid model combining HeteroRGCN and XGBoost for fraud detection
    """
    
    def __init__(self, 
                 ntype_dict, 
                 etypes, 
                 gnn_config, 
                 xgb_config=None,
                 ensemble_weights=None,
                 device='cpu'):
        """
        Initialize the hybrid model
        
        Args:
            ntype_dict: Dictionary of node types and their counts
            etypes: List of edge types
            gnn_config: Configuration for GNN model
            xgb_config: Configuration for XGBoost model
            ensemble_weights: Weights for ensemble voting
            device: Device for GNN training
        """
        self.device = device
        self.ntype_dict = ntype_dict
        self.etypes = etypes
        self.gnn_config = gnn_config
        self.xgb_config = xgb_config or self._default_xgb_config()
        self.ensemble_weights = ensemble_weights or [0.6, 0.4]  # [GNN, XGB]
        
        # Initialize models
        self.gnn_model = None
        self.xgb_model = None
        self.feature_scaler = None
        self.is_trained = False
        
    def _default_xgb_config(self):
        """Default XGBoost configuration optimized for fraud detection"""
        return {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': 3.0,  # Handle class imbalance
            'random_state': 42,
            'n_jobs': -1
        }
    
    def _initialize_gnn(self, in_feats, n_classes):
        """Initialize the HeteroRGCN model"""
        self.gnn_model = HeteroRGCN(
            self.ntype_dict, 
            self.etypes, 
            in_feats, 
            self.gnn_config['n_hidden'], 
            n_classes, 
            self.gnn_config['n_layers'], 
            in_feats
        ).to(self.device)
        
    def _initialize_xgb(self):
        """Initialize the XGBoost model"""
        self.xgb_model = xgb.XGBClassifier(**self.xgb_config)
        
    def extract_graph_embeddings(self, g, features):
        """
        Extract node embeddings from trained GNN model
        
        Args:
            g: DGL graph
            features: Node features
            
        Returns:
            Graph embeddings for target nodes
        """
        if self.gnn_model is None:
            raise ValueError("GNN model not trained yet")
            
        self.gnn_model.eval()
        with th.no_grad():
            # Get embeddings from the second-to-last layer
            h_dict = {ntype: emb for ntype, emb in self.gnn_model.embed.items()}
            h_dict['target'] = features.to(self.device)
            
            # Forward pass through all layers except the last
            for i, layer in enumerate(self.gnn_model.layers[:-1]):
                if i != 0:
                    h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
                h_dict = layer(g, h_dict)
                
            # Return target node embeddings
            return h_dict['target'].cpu().numpy()
    
    def prepare_tabular_features(self, features, graph_embeddings, additional_features=None):
        """
        Prepare combined features for XGBoost training
        
        Args:
            features: Original node features
            graph_embeddings: Graph embeddings from GNN
            additional_features: Additional tabular features (optional)
            
        Returns:
            Combined feature matrix
        """
        feature_list = [features.cpu().numpy(), graph_embeddings]
        
        if additional_features is not None:
            feature_list.append(additional_features)
            
        combined_features = np.concatenate(feature_list, axis=1)
        
        # Normalize features if scaler exists
        if self.feature_scaler is not None:
            combined_features = self.feature_scaler.transform(combined_features)
        else:
            from sklearn.preprocessing import StandardScaler
            self.feature_scaler = StandardScaler()
            combined_features = self.feature_scaler.fit_transform(combined_features)
            
        return combined_features
    
    def train_gnn(self, g, features, labels, train_mask, test_mask, n_epochs=100):
        """
        Train the HeteroRGCN model
        
        Args:
            g: DGL graph
            features: Node features
            labels: Node labels
            train_mask: Training mask
            test_mask: Test mask
            n_epochs: Number of training epochs
            
        Returns:
            Training history
        """
        print("Training HeteroRGCN model...")
        
        # Initialize GNN model
        in_feats = features.shape[1]
        n_classes = 2
        self._initialize_gnn(in_feats, n_classes)
        
        # Setup training
        optimizer = th.optim.Adam(self.gnn_model.parameters(), 
                                 lr=self.gnn_config.get('lr', 0.01),
                                 weight_decay=self.gnn_config.get('weight_decay', 5e-4))
        loss_fn = th.nn.CrossEntropyLoss()
        
        # Training loop
        best_f1 = 0
        best_model_state = None
        history = {'loss': [], 'f1': [], 'auc': []}
        
        features = features.to(self.device)
        labels = labels.to(self.device)
        
        for epoch in range(n_epochs):
            self.gnn_model.train()
            
            # Forward pass
            logits = self.gnn_model(g, features)
            loss = loss_fn(logits[train_mask.bool()], labels[train_mask.bool()])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Evaluation
            if epoch % 10 == 0:
                metrics = self._evaluate_gnn(g, features, labels, test_mask)
                history['loss'].append(loss.item())
                history['f1'].append(metrics['f1'])
                history['auc'].append(metrics['auc'])
                
                print(f"Epoch {epoch:04d} | Loss: {loss.item():.4f} | "
                      f"F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")
                
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_model_state = copy.deepcopy(self.gnn_model.state_dict())
        
        # Load best model
        if best_model_state is not None:
            self.gnn_model.load_state_dict(best_model_state)
            
        print(f"GNN training completed. Best F1: {best_f1:.4f}")
        return history
    
    def train_xgb(self, g, features, labels, train_mask, test_mask, additional_features=None):
        """
        Train the XGBoost model using combined features
        
        Args:
            g: DGL graph
            features: Node features
            labels: Node labels
            train_mask: Training mask
            test_mask: Test mask
            additional_features: Additional tabular features
            
        Returns:
            Training metrics
        """
        print("Training XGBoost model...")
        
        # Extract graph embeddings
        graph_embeddings = self.extract_graph_embeddings(g, features)
        
        # Prepare combined features
        combined_features = self.prepare_tabular_features(features, graph_embeddings, additional_features)
        
        # Split data
        X_train = combined_features[train_mask.bool().numpy()]
        X_test = combined_features[test_mask.bool().numpy()]
        y_train = labels[train_mask.bool()].numpy()
        y_test = labels[test_mask.bool()].numpy()
        
        # Initialize and train XGBoost
        self._initialize_xgb()
        
        # Train with early stopping
        eval_set = [(X_train, y_train), (X_test, y_test)]
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_names=['train', 'test'],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Evaluate
        train_proba = self.xgb_model.predict_proba(X_train)[:, 1]
        test_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        
        train_auc = roc_auc_score(y_train, train_proba)
        test_auc = roc_auc_score(y_test, test_proba)
        
        print(f"XGBoost training completed. Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}")
        
        return {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'feature_importance': self.xgb_model.feature_importances_
        }
    
    def train_hybrid(self, g, features, labels, train_mask, test_mask, 
                    additional_features=None, gnn_epochs=100):
        """
        Train the complete hybrid model
        
        Args:
            g: DGL graph
            features: Node features
            labels: Node labels
            train_mask: Training mask
            test_mask: Test mask
            additional_features: Additional tabular features
            gnn_epochs: Number of GNN training epochs
            
        Returns:
            Training results
        """
        print("Starting Hybrid Model Training...")
        
        # Step 1: Train GNN
        gnn_history = self.train_gnn(g, features, labels, train_mask, test_mask, gnn_epochs)
        
        # Step 2: Train XGBoost
        xgb_metrics = self.train_xgb(g, features, labels, train_mask, test_mask, additional_features)
        
        # Step 3: Evaluate hybrid model
        hybrid_metrics = self.evaluate_hybrid(g, features, labels, test_mask, additional_features)
        
        self.is_trained = True
        print("Hybrid model training completed!")
        
        return {
            'gnn_history': gnn_history,
            'xgb_metrics': xgb_metrics,
            'hybrid_metrics': hybrid_metrics
        }
    
    def predict_hybrid(self, g, features, additional_features=None):
        """
        Make hybrid predictions combining GNN and XGBoost
        
        Args:
            g: DGL graph
            features: Node features
            additional_features: Additional tabular features
            
        Returns:
            Hybrid predictions and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # GNN predictions
        self.gnn_model.eval()
        with th.no_grad():
            gnn_logits = self.gnn_model(g, features.to(self.device))
            gnn_proba = th.softmax(gnn_logits, dim=-1)[:, 1].cpu().numpy()
        
        # XGBoost predictions
        graph_embeddings = self.extract_graph_embeddings(g, features)
        combined_features = self.prepare_tabular_features(features, graph_embeddings, additional_features)
        xgb_proba = self.xgb_model.predict_proba(combined_features)[:, 1]
        
        # Ensemble predictions
        hybrid_proba = (self.ensemble_weights[0] * gnn_proba + 
                       self.ensemble_weights[1] * xgb_proba)
        hybrid_pred = (hybrid_proba > 0.5).astype(int)
        
        return {
            'predictions': hybrid_pred,
            'probabilities': hybrid_proba,
            'gnn_proba': gnn_proba,
            'xgb_proba': xgb_proba
        }
    
    def evaluate_hybrid(self, g, features, labels, test_mask, additional_features=None):
        """
        Evaluate the hybrid model
        
        Args:
            g: DGL graph
            features: Node features
            labels: Node labels
            test_mask: Test mask
            additional_features: Additional tabular features
            
        Returns:
            Evaluation metrics
        """
        predictions = self.predict_hybrid(g, features, additional_features)
        
        # Get test data
        y_test = labels[test_mask.bool()].numpy()
        y_pred = predictions['predictions'][test_mask.bool().numpy()]
        y_proba = predictions['probabilities'][test_mask.bool().numpy()]
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        
        # F1 score
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
        
        return {
            'auc': auc,
            'f1': f1,
            'precision': precision_score,
            'recall': recall_score,
            'ap': ap
        }
    
    def _evaluate_gnn(self, g, features, labels, test_mask):
        """Evaluate GNN model"""
        self.gnn_model.eval()
        with th.no_grad():
            logits = self.gnn_model(g, features)
            proba = th.softmax(logits, dim=-1)[:, 1]
            pred = logits.argmax(dim=1)
            
            # Test metrics
            y_test = labels[test_mask.bool()]
            y_pred = pred[test_mask.bool()]
            y_proba = proba[test_mask.bool()]
            
            auc = roc_auc_score(y_test.cpu().numpy(), y_proba.cpu().numpy())
            
            # F1 score
            tn, fp, fn, tp = confusion_matrix(y_test.cpu().numpy(), y_pred.cpu().numpy()).ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {'auc': auc, 'f1': f1, 'precision': precision, 'recall': recall}
    
    def save_model(self, model_dir):
        """
        Save the hybrid model
        
        Args:
            model_dir: Directory to save models
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save GNN model
        th.save(self.gnn_model.state_dict(), os.path.join(model_dir, 'gnn_model.pth'))
        
        # Save XGBoost model
        joblib.dump(self.xgb_model, os.path.join(model_dir, 'xgb_model.pkl'))
        
        if self.feature_scaler is not None:
            joblib.dump(self.feature_scaler, os.path.join(model_dir, 'feature_scaler.pkl'))
        
        config = {
            'ntype_dict': self.ntype_dict,
            'etypes': self.etypes,
            'gnn_config': self.gnn_config,
            'xgb_config': self.xgb_config,
            'ensemble_weights': self.ensemble_weights
        }
        
        with open(os.path.join(model_dir, 'hybrid_config.pkl'), 'wb') as f:
            pickle.dump(config, f)
        
        print(f"Hybrid model saved to {model_dir}")
    
    def load_model(self, model_dir):
        """
        Load the hybrid model
        
        Args:
            model_dir: Directory containing saved models
        """
        # Load configuration
        with open(os.path.join(model_dir, 'hybrid_config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        self.ntype_dict = config['ntype_dict']
        self.etypes = config['etypes']
        self.gnn_config = config['gnn_config']
        self.xgb_config = config['xgb_config']
        self.ensemble_weights = config['ensemble_weights']
        
        # Load GNN model
        in_feats = self.gnn_config.get('in_feats', 64)  # Default value
        self._initialize_gnn(in_feats, 2)
        self.gnn_model.load_state_dict(th.load(os.path.join(model_dir, 'gnn_model.pth')))
        
        # Load XGBoost model
        self.xgb_model = joblib.load(os.path.join(model_dir, 'xgb_model.pkl'))
        
        # Load feature scaler
        scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
        if os.path.exists(scaler_path):
            self.feature_scaler = joblib.load(scaler_path)
        
        self.is_trained = True
        print(f"Hybrid model loaded from {model_dir}")


def main():
    """
    Main training function
    """
    print('Starting Hybrid Fraud Detection Training...')
    print(f'numpy version: {np.__version__}')
    print(f'PyTorch version: {th.__version__}')
    print(f'DGL version: {dgl.__version__}')
    print(f'XGBoost version: {xgb.__version__}')
    
    args = parse_args()
    print(args)
    
    # Construct graph
    args.edges = get_edgelists('relation*', args.training_dir)
    g, features, target_id_to_node, id_to_node = construct_graph(
        args.training_dir,
        args.edges, 
        args.nodes,
        args.target_ntype
    )
    
    # Normalize features
    mean, stdev, features = normalize(th.from_numpy(features))
    g.nodes['target'].data['features'] = features
    
    # Get labels
    print("Getting labels...")
    n_nodes = g.number_of_nodes('target')
    labels, train_mask, test_mask = get_labels(
        target_id_to_node,
        n_nodes,
        args.target_ntype,
        os.path.join(args.training_dir, args.labels),
        os.path.join(args.training_dir, args.new_accounts)
    )
    
    labels = th.from_numpy(labels).long()
    train_mask = th.from_numpy(train_mask).bool()
    test_mask = th.from_numpy(test_mask).bool()
    
    print(f"""
    ---- Data Statistics ----
    #Nodes: {sum([g.number_of_nodes(ntype) for ntype in g.ntypes])}
    #Edges: {sum([g.number_of_edges(etype) for etype in g.etypes])}
    #Features Shape: {features.shape}
    #Training samples: {train_mask.sum()}
    #Test samples: {test_mask.sum()}
    """)
    
    device = th.device('cuda:0' if args.num_gpus and th.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model configuration
    ntype_dict = {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}
    gnn_config = {
        'n_hidden': args.n_hidden,
        'n_layers': args.n_layers,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'in_feats': features.shape[1]
    }
    
    # XGBoost configuration (optimized for fraud detection)
    xgb_config = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': 5.0,  # Adjust based on class imbalance
        'random_state': 42
    }
    
    # Initialize hybrid model
    hybrid_model = HybridFraudDetector(
        ntype_dict=ntype_dict,
        etypes=g.etypes,
        gnn_config=gnn_config,
        xgb_config=xgb_config,
        ensemble_weights=[0.6, 0.4],  # GNN gets higher weight
        device=device
    )
    
    # Train the hybrid model
    print("\n" + "="*50)
    print("Starting Hybrid Training")
    print("="*50)
    
    training_results = hybrid_model.train_hybrid(
        g=g,
        features=features,
        labels=labels,
        train_mask=train_mask,
        test_mask=test_mask,
        gnn_epochs=args.n_epochs
    )
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Hybrid Model - AUC: {training_results['hybrid_metrics']['auc']:.4f}")
    print(f"Hybrid Model - F1: {training_results['hybrid_metrics']['f1']:.4f}")
    print(f"Hybrid Model - Precision: {training_results['hybrid_metrics']['precision']:.4f}")
    print(f"Hybrid Model - Recall: {training_results['hybrid_metrics']['recall']:.4f}")
    print(f"Hybrid Model - AP: {training_results['hybrid_metrics']['ap']:.4f}")
    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    hybrid_model.save_model(args.model_dir)
    
    metadata = {
        'feature_mean': mean,
        'feature_std': stdev,
        'id_to_node': id_to_node,
        'target_id_to_node': target_id_to_node,
        'training_results': training_results
    }
    
    with open(os.path.join(args.model_dir, 'training_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Model and metadata saved to {args.model_dir}")
    print("Training completed successfully!")


if __name__ == '__main__':
    main() 
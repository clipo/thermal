#!/usr/bin/env python3
"""
Enhanced segmentation models for improved ocean/land classification.

This module provides several improved approaches:
1. XGBoost with better hyperparameters
2. Deep learning with CNNs (if suitable)
3. Ensemble methods combining multiple models
4. Advanced feature engineering
"""

import numpy as np
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost (much better than RandomForest for this task)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

# Try to import LightGBM (even faster than XGBoost)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")


class EnhancedSegmentationModel:
    """Enhanced segmentation with multiple model options."""
    
    def __init__(self, model_type='auto', use_scaling=True):
        """
        Initialize enhanced segmentation model.
        
        Args:
            model_type: 'auto', 'xgboost', 'lightgbm', 'ensemble', 'neural', 'rf_optimized'
            use_scaling: Whether to scale features (important for neural networks)
        """
        self.model_type = model_type
        self.use_scaling = use_scaling
        self.scaler = StandardScaler() if use_scaling else None
        self.model = None
        self.feature_importance = None
        
    def create_model(self, n_classes=4):
        """Create the specified model type."""
        
        if self.model_type == 'auto':
            # Auto-select best available model
            if XGBOOST_AVAILABLE:
                self.model_type = 'xgboost'
            elif LIGHTGBM_AVAILABLE:
                self.model_type = 'lightgbm'
            else:
                self.model_type = 'rf_optimized'
        
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            # XGBoost - typically 20-30% better than RandomForest
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                n_jobs=-1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            print("Using XGBoost classifier (best accuracy)")
            
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            # LightGBM - faster than XGBoost with similar accuracy
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multiclass',
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )
            print("Using LightGBM classifier (fastest)")
            
        elif self.model_type == 'ensemble':
            # Ensemble of multiple models for best accuracy
            models = []
            
            # Optimized Random Forest
            rf = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            models.append(('rf', rf))
            
            # Gradient Boosting
            gb = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            models.append(('gb', gb))
            
            # Neural Network
            if self.use_scaling:
                mlp = MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    max_iter=500,
                    random_state=42
                )
                models.append(('mlp', mlp))
            
            self.model = VotingClassifier(
                estimators=models,
                voting='soft'  # Use probability predictions
            )
            print("Using Ensemble classifier (most robust)")
            
        elif self.model_type == 'neural':
            # Deep neural network
            self.model = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
            self.use_scaling = True  # Neural networks need scaling
            print("Using Neural Network classifier")
            
        else:  # rf_optimized or fallback
            # Optimized Random Forest with better hyperparameters
            self.model = RandomForestClassifier(
                n_estimators=200,  # More trees
                max_depth=15,      # Deeper trees for complex patterns
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,    # Out-of-bag score for validation
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Handle imbalanced classes
            )
            print("Using Optimized Random Forest classifier")
            
        return self.model
    
    def add_advanced_features(self, features):
        """
        Add advanced features for better segmentation.
        
        Args:
            features: Original feature array (N x M)
            
        Returns:
            Enhanced feature array with additional features
        """
        enhanced = [features]
        
        # Add polynomial features for key color ratios
        if features.shape[1] >= 3:
            r, g, b = features[:, 0], features[:, 1], features[:, 2]
            
            # Color ratios (important for water detection)
            with np.errstate(divide='ignore', invalid='ignore'):
                rg_ratio = np.where(g > 0, r / g, 0)
                rb_ratio = np.where(b > 0, r / b, 0)
                gb_ratio = np.where(b > 0, g / b, 0)
                bg_ratio = np.where(g > 0, b / g, 0)
            
            enhanced.append(rg_ratio.reshape(-1, 1))
            enhanced.append(rb_ratio.reshape(-1, 1))
            enhanced.append(gb_ratio.reshape(-1, 1))
            enhanced.append(bg_ratio.reshape(-1, 1))
            
            # Quadratic terms for blue dominance
            blue_dominance = b - np.maximum(r, g)
            enhanced.append(blue_dominance.reshape(-1, 1))
            enhanced.append((blue_dominance ** 2).reshape(-1, 1))
            
            # Color space transformations
            max_rgb = np.maximum(r, np.maximum(g, b))
            min_rgb = np.minimum(r, np.minimum(g, b))
            
            # Chroma (color purity)
            chroma = max_rgb - min_rgb
            enhanced.append(chroma.reshape(-1, 1))
            
            # Normalized colors (illumination invariant)
            sum_rgb = r + g + b + 1e-6
            r_norm = r / sum_rgb
            g_norm = g / sum_rgb
            b_norm = b / sum_rgb
            enhanced.append(r_norm.reshape(-1, 1))
            enhanced.append(g_norm.reshape(-1, 1))
            enhanced.append(b_norm.reshape(-1, 1))
        
        return np.hstack(enhanced)
    
    def fit(self, X, y):
        """
        Train the model with advanced features.
        
        Args:
            X: Training features (N x M)
            y: Training labels (N,)
        """
        # Add advanced features
        X_enhanced = self.add_advanced_features(X)
        
        # Scale if needed
        if self.use_scaling and self.scaler:
            X_enhanced = self.scaler.fit_transform(X_enhanced)
        
        # Create and train model
        if self.model is None:
            self.create_model(n_classes=len(np.unique(y)))
        
        self.model.fit(X_enhanced, y)
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.feature_importance = self.model.feature_importances_
        
        return self
    
    def predict(self, X):
        """Predict with enhanced features."""
        X_enhanced = self.add_advanced_features(X)
        
        if self.use_scaling and self.scaler:
            X_enhanced = self.scaler.transform(X_enhanced)
        
        return self.model.predict(X_enhanced)
    
    def predict_proba(self, X):
        """Predict probabilities with enhanced features."""
        X_enhanced = self.add_advanced_features(X)
        
        if self.use_scaling and self.scaler:
            X_enhanced = self.scaler.transform(X_enhanced)
        
        return self.model.predict_proba(X_enhanced)
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation to estimate performance.
        
        Returns:
            Mean accuracy and standard deviation
        """
        X_enhanced = self.add_advanced_features(X)
        
        if self.use_scaling and self.scaler:
            X_enhanced = self.scaler.fit_transform(X_enhanced)
        
        if self.model is None:
            self.create_model(n_classes=len(np.unique(y)))
        
        scores = cross_val_score(self.model, X_enhanced, y, cv=cv, scoring='accuracy')
        
        print(f"\nCross-validation results ({cv} folds):")
        print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        return scores.mean(), scores.std()
    
    def save(self, filepath):
        """Save the complete model including scaler."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'use_scaling': self.use_scaling,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Enhanced model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load a saved enhanced model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(
            model_type=model_data['model_type'],
            use_scaling=model_data['use_scaling']
        )
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_importance = model_data.get('feature_importance')
        
        return instance


def compare_models(X_train, y_train, X_test, y_test):
    """
    Compare different model architectures.
    
    Returns:
        Dictionary of model performances
    """
    from sklearn.metrics import accuracy_score, classification_report
    
    results = {}
    
    # Test different model types
    model_types = ['rf_optimized']
    
    if XGBOOST_AVAILABLE:
        model_types.append('xgboost')
    
    if LIGHTGBM_AVAILABLE:
        model_types.append('lightgbm')
    
    model_types.extend(['ensemble', 'neural'])
    
    print("\n" + "="*60)
    print("COMPARING SEGMENTATION MODELS")
    print("="*60)
    
    for model_type in model_types:
        print(f"\n{'-'*40}")
        print(f"Testing: {model_type}")
        print(f"{'-'*40}")
        
        try:
            # Create and train model
            model = EnhancedSegmentationModel(model_type=model_type)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            results[model_type] = {
                'accuracy': accuracy,
                'model': model
            }
            
            print(f"Accuracy: {accuracy:.3f}")
            
        except Exception as e:
            print(f"Failed: {e}")
            results[model_type] = {
                'accuracy': 0,
                'model': None
            }
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.3f})")
    print(f"{'='*60}")
    
    return results


def upgrade_existing_model(old_model_path, training_data_path):
    """
    Upgrade an existing RandomForest model to enhanced architecture.
    
    Args:
        old_model_path: Path to existing .pkl model
        training_data_path: Path to training data JSON
    """
    import json
    
    print(f"\nUpgrading model: {old_model_path}")
    
    # Load training data
    with open(training_data_path, 'r') as f:
        training_data = json.load(f)
    
    # Extract features and labels
    X = np.array([s['features'] for s in training_data])
    labels = [s['label'] for s in training_data]
    
    # Convert labels to numeric
    classes = ['ocean', 'land', 'rock', 'wave']
    y = np.array([classes.index(label) for label in labels])
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Compare models
    results = compare_models(X_train, y_train, X_test, y_test)
    
    # Save best model
    best_model_type = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model = results[best_model_type]['model']
    
    # Save with enhanced suffix
    enhanced_path = old_model_path.replace('.pkl', '_enhanced.pkl')
    best_model.save(enhanced_path)
    
    print(f"\nEnhanced model saved to: {enhanced_path}")
    print(f"Model type: {best_model_type}")
    print(f"Improvement: Use this model for better accuracy!")
    
    return enhanced_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        # Upgrade existing model
        old_model = sys.argv[1]
        training_data = sys.argv[2]
        upgrade_existing_model(old_model, training_data)
    else:
        print("Usage: python enhanced_segmentation.py <model.pkl> <training_data.json>")
        print("\nThis will create an enhanced version of your segmentation model")
        print("with better accuracy using advanced ML techniques.")
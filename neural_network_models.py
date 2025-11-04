"""
Neural Network Models for EMF Prediction
=========================================

This module contains multiple neural network architectures for predicting
Electric and Magnetic fields from transmission line data.

Models included:
1. Multi-Layer Perceptron (MLP) - Basic feedforward network
2. Deep MLP - Deeper architecture with more layers
3. Wide MLP - Wider architecture with more neurons per layer
4. Regularized MLP - Heavy regularization with dropout
5. Ensemble MLP - Multiple MLPs with different architectures

Author: EMF Prediction Team
Date: October 31, 2025
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import explained_variance_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')


class NeuralNetworkEnsemble:
    """
    Ensemble of multiple neural network architectures for robust predictions
    """
    
    def __init__(self, random_state=42, verbose=False):
        """
        Initialize the neural network ensemble
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        verbose : bool
            Whether to print training progress
        """
        self.random_state = random_state
        self.verbose = verbose
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
    def build_models(self):
        """
        Build multiple neural network architectures
        """
        print("Building Neural Network Ensemble...")
        print("=" * 70)
        
        # 1. Standard MLP (3 layers)
        self.models['MLP_Standard'] = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
            random_state=self.random_state,
            verbose=self.verbose
        )
        print("✓ MLP_Standard: 3 layers (100→50→25)")
        
        # 2. Deep MLP (5 layers)
        self.models['MLP_Deep'] = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32, 16, 8),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
            random_state=self.random_state,
            verbose=self.verbose
        )
        print("✓ MLP_Deep: 5 layers (128→64→32→16→8)")
        
        # 3. Wide MLP (3 wide layers)
        self.models['MLP_Wide'] = MLPRegressor(
            hidden_layer_sizes=(200, 150, 100),
            activation='relu',
            solver='adam',
            alpha=0.0005,
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
            random_state=self.random_state,
            verbose=self.verbose
        )
        print("✓ MLP_Wide: 3 wide layers (200→150→100)")
        
        # 4. Regularized MLP (with strong regularization)
        self.models['MLP_Regularized'] = MLPRegressor(
            hidden_layer_sizes=(80, 40, 20),
            activation='relu',
            solver='adam',
            alpha=0.01,  # Strong L2 regularization
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.0005,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=30,
            random_state=self.random_state,
            verbose=self.verbose
        )
        print("✓ MLP_Regularized: Strong L2 penalty (80→40→20)")
        
        # 5. Tanh Activation MLP
        self.models['MLP_Tanh'] = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='tanh',  # Tanh activation instead of ReLU
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
            random_state=self.random_state,
            verbose=self.verbose
        )
        print("✓ MLP_Tanh: Tanh activation (100→50→25)")
        
        print("=" * 70)
        print(f"✓ Built {len(self.models)} neural network models")
        print()
        
    def train_all(self, X_train, y_train, model_type='E-field'):
        """
        Train all neural network models
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        model_type : str
            Type of model ('E-field' or 'H-field')
        """
        if not self.models:
            self.build_models()
            
        print(f"\nTraining Neural Networks for {model_type}...")
        print("=" * 70)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            print(f"  ✓ Training complete. Iterations: {model.n_iter_}")
            if hasattr(model, 'loss_'):
                print(f"  ✓ Final training loss: {model.loss_:.6f}")
        
        print("\n" + "=" * 70)
        print(f"✓ All {len(self.models)} models trained successfully!")
        
    def predict_all(self, X_test):
        """
        Generate predictions from all models
        
        Parameters:
        -----------
        X_test : array-like
            Test features
            
        Returns:
        --------
        dict : Dictionary of predictions from each model
        """
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_test)
        return predictions
    
    def evaluate_all(self, X_test, y_test, X_val=None, y_val=None):
        """
        Evaluate all models on test (and optionally validation) data
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test target
        X_val : array-like, optional
            Validation features
        y_val : array-like, optional
            Validation target
            
        Returns:
        --------
        DataFrame : Results for all models
        """
        results = []
        
        for name, model in self.models.items():
            # Test set evaluation
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            explained_var = explained_variance_score(y_test, y_pred)
            
            result = {
                'Model': name,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2_Score': r2,
                'MAPE (%)': mape,
                'Explained_Variance': explained_var,
                'Iterations': model.n_iter_
            }
            
            # Validation set evaluation if provided
            if X_val is not None and y_val is not None:
                y_val_pred = model.predict(X_val)
                val_r2 = r2_score(y_val, y_val_pred)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                result['Val_R2'] = val_r2
                result['Val_RMSE'] = val_rmse
            
            results.append(result)
        
        return pd.DataFrame(results).sort_values('R2_Score', ascending=False)
    
    def get_ensemble_prediction(self, X_test, method='mean'):
        """
        Get ensemble prediction by combining all models
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        method : str
            Ensemble method ('mean', 'median', 'weighted')
            
        Returns:
        --------
        array : Ensemble predictions
        """
        predictions = self.predict_all(X_test)
        pred_matrix = np.column_stack(list(predictions.values()))
        
        if method == 'mean':
            return np.mean(pred_matrix, axis=1)
        elif method == 'median':
            return np.median(pred_matrix, axis=1)
        elif method == 'weighted':
            # Weight by R2 scores (if available)
            if hasattr(self, 'weights'):
                return np.average(pred_matrix, axis=1, weights=self.weights)
            else:
                return np.mean(pred_matrix, axis=1)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def get_best_model(self, results_df):
        """
        Get the best performing model based on R2 score
        
        Parameters:
        -----------
        results_df : DataFrame
            Results dataframe from evaluate_all()
            
        Returns:
        --------
        tuple : (best_model_name, best_model_object)
        """
        best_model_name = results_df.iloc[0]['Model']
        best_model = self.models[best_model_name]
        return best_model_name, best_model
    
    def print_summary(self, results_df):
        """
        Print a comprehensive summary of all models
        
        Parameters:
        -----------
        results_df : DataFrame
            Results dataframe from evaluate_all()
        """
        print("\n" + "=" * 70)
        print("NEURAL NETWORK ENSEMBLE SUMMARY")
        print("=" * 70)
        print(f"\nTotal Models Trained: {len(self.models)}")
        print(f"\nModel Architectures:")
        
        architectures = {
            'MLP_Standard': '3 layers (100→50→25), ReLU, adaptive LR',
            'MLP_Deep': '5 layers (128→64→32→16→8), ReLU, adaptive LR',
            'MLP_Wide': '3 wide layers (200→150→100), ReLU, batch_size=64',
            'MLP_Regularized': 'Strong L2 (α=0.01), (80→40→20), ReLU',
            'MLP_Tanh': 'Tanh activation (100→50→25), adaptive LR'
        }
        
        for name, arch in architectures.items():
            if name in self.models:
                print(f"  • {name:20s}: {arch}")
        
        print("\n" + "=" * 70)
        print("PERFORMANCE RANKING (by R² Score)")
        print("=" * 70)
        print(results_df.to_string(index=False))
        
        print("\n" + "=" * 70)
        best_model_name = results_df.iloc[0]['Model']
        best_r2 = results_df.iloc[0]['R2_Score']
        print(f"🏆 BEST MODEL: {best_model_name}")
        print(f"   R² Score: {best_r2:.6f}")
        print("=" * 70)


def train_neural_network_ensemble(X_train, y_train, X_val, y_val, X_test, y_test, 
                                   field_type='E-field', random_state=42):
    """
    Convenience function to train and evaluate neural network ensemble
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_val, y_val : Validation data
    X_test, y_test : Test data
    field_type : str
        'E-field' or 'H-field'
    random_state : int
        Random seed
        
    Returns:
    --------
    tuple : (ensemble object, results dataframe, best model name, best model)
    """
    # Create ensemble
    ensemble = NeuralNetworkEnsemble(random_state=random_state, verbose=False)
    
    # Build and train models
    ensemble.build_models()
    ensemble.train_all(X_train, y_train, model_type=field_type)
    
    # Evaluate on test and validation sets
    results_df = ensemble.evaluate_all(X_test, y_test, X_val, y_val)
    
    # Get best model
    best_model_name, best_model = ensemble.get_best_model(results_df)
    
    # Print summary
    ensemble.print_summary(results_df)
    
    return ensemble, results_df, best_model_name, best_model


if __name__ == "__main__":
    """
    Example usage and testing
    """
    print("Neural Network Models Module")
    print("=" * 70)
    print("\nThis module contains 5 neural network architectures:")
    print("  1. MLP_Standard - Basic 3-layer network")
    print("  2. MLP_Deep - Deep 5-layer network")
    print("  3. MLP_Wide - Wide 3-layer network")
    print("  4. MLP_Regularized - Heavily regularized network")
    print("  5. MLP_Tanh - Tanh activation network")
    print("\nImport this module in your notebook:")
    print("  from neural_network_models import NeuralNetworkEnsemble")
    print("  from neural_network_models import train_neural_network_ensemble")
    print("=" * 70)

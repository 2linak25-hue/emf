"""
Quick Reference Guide - Neural Network Ensemble Module
======================================================

QUICK START
-----------

1. Import the module:
   from neural_network_models import train_neural_network_ensemble

2. Train ensemble (one line):
   ensemble, results, best_name, best_model = train_neural_network_ensemble(
       X_train, y_train, X_val, y_val, X_test, y_test, 
       field_type='E-field', random_state=42
   )

3. View results:
   print(results)

4. Make predictions:
   predictions = best_model.predict(X_new)


AVAILABLE MODELS
----------------

1. MLP_Standard     → Balanced (100→50→25, ReLU)
2. MLP_Deep         → Deep learning (128→64→32→16→8, ReLU)
3. MLP_Wide         → Wide layers (200→150→100, ReLU)
4. MLP_Regularized  → Anti-overfitting (80→40→20, strong L2)
5. MLP_Tanh         → Alternative activation (100→50→25, Tanh)


COMMON TASKS
------------

✓ Get best model:
  best_name, best_model = ensemble.get_best_model(results)

✓ Ensemble prediction (average all models):
  pred = ensemble.get_ensemble_prediction(X_test, method='mean')

✓ Individual predictions:
  predictions_dict = ensemble.predict_all(X_test)

✓ Print summary:
  ensemble.print_summary(results)


METRICS RETURNED
----------------

- R2_Score          → R² score (higher is better, max 1.0)
- RMSE              → Root mean squared error (lower is better)
- MAE               → Mean absolute error (lower is better)
- MSE               → Mean squared error (lower is better)
- MAPE (%)          → Mean absolute percentage error
- Explained_Variance → Variance explained
- Iterations        → Training iterations until convergence
- Val_R2            → Validation R² score
- Val_RMSE          → Validation RMSE


CHOOSING THE RIGHT ARCHITECTURE
--------------------------------

Use MLP_Standard if:
  ✓ You want a good all-around model
  ✓ Data has moderate complexity

Use MLP_Deep if:
  ✓ Data has complex non-linear patterns
  ✓ Many features with interactions
  ✓ You have enough training data (>1000 samples)

Use MLP_Wide if:
  ✓ High-dimensional data
  ✓ Many input features (>20)
  ✓ Feature interactions are important

Use MLP_Regularized if:
  ✓ Overfitting is a concern
  ✓ Validation R² much lower than training R²
  ✓ Limited training data

Use MLP_Tanh if:
  ✓ ReLU models aren't performing well
  ✓ Data is well-normalized
  ✓ Want different non-linearity


TROUBLESHOOTING
---------------

Problem: Low accuracy
Solution: 
  1. Check if data is normalized
  2. Try ensemble prediction instead of single model
  3. Increase max_iter in the module

Problem: Overfitting (high train R², low val R²)
Solution:
  1. Use MLP_Regularized
  2. Increase alpha parameter
  3. Reduce number of layers/neurons

Problem: Underfitting (low train R² and low val R²)
Solution:
  1. Use MLP_Deep or MLP_Wide
  2. Increase max_iter
  3. Check data quality and normalization

Problem: Inconsistent results
Solution:
  1. Use ensemble prediction (averages all models)
  2. Set random_state for reproducibility
  3. Check validation set size (should be 15-20% of train)


ADVANCED USAGE
--------------

Manual training:
  ensemble = NeuralNetworkEnsemble(random_state=42)
  ensemble.build_models()
  ensemble.train_all(X_train, y_train)
  results = ensemble.evaluate_all(X_test, y_test, X_val, y_val)

Access individual models:
  mlp_deep = ensemble.models['MLP_Deep']
  prediction = mlp_deep.predict(X_new)

Weighted ensemble:
  ensemble.weights = [r2_score1, r2_score2, ...]  # Set weights
  pred = ensemble.get_ensemble_prediction(X_test, method='weighted')


INTEGRATION WITH MAIN NOTEBOOK
-------------------------------

The module is automatically imported in cell 12 of the main notebook:

  from neural_network_models import train_neural_network_ensemble

Then used in cells 13 (E-field) and 17 (H-field) for training.

Results are integrated with other models (RF, XGBoost, LightGBM, SVR)
for comprehensive comparison.


BEST PRACTICES
--------------

1. ✓ Always use validation set for model selection
2. ✓ Compare all architectures before choosing
3. ✓ Use ensemble prediction for robustness
4. ✓ Monitor training iterations (early stopping)
5. ✓ Normalize input data before training
6. ✓ Compare NNs with tree-based models
7. ✓ Save best model for deployment


PERFORMANCE EXPECTATIONS
-------------------------

Expected R² scores on EMF data:
- Good models:     R² > 0.90
- Excellent models: R² > 0.95
- Perfect fit:     R² ≈ 1.00

Typical training time per model: 5-30 seconds
Total ensemble training: 30-180 seconds (5 models)


EXPORT RESULTS
--------------

Save results to CSV:
  results.to_csv('nn_ensemble_results.csv', index=False)

Save best model:
  import joblib
  joblib.dump(best_model, 'best_nn_model.pkl')

Load saved model:
  loaded_model = joblib.load('best_nn_model.pkl')


COMMON PARAMETERS
-----------------

hidden_layer_sizes  → Tuple of layer sizes (e.g., (100, 50, 25))
activation          → 'relu', 'tanh', 'logistic'
solver              → 'adam' (default), 'sgd', 'lbfgs'
alpha               → L2 regularization (0.0001 to 0.01)
learning_rate       → 'adaptive', 'constant', 'invscaling'
max_iter            → Maximum iterations (default: 1000)
early_stopping      → True (recommended)
validation_fraction → Validation set size (default: 0.15)
n_iter_no_change    → Patience for early stopping (default: 50)
random_state        → Seed for reproducibility (default: 42)


CONTACT & SUPPORT
-----------------

For questions or issues:
1. Check this quick reference
2. Read NEURAL_NETWORK_README.md
3. Review neural_network_models.py docstrings
4. Check main notebook examples

---
Last Updated: October 31, 2025
Version: 1.0
"""

if __name__ == "__main__":
    print(__doc__)

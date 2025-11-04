# Neural Network Ensemble for EMF Prediction

## Overview
This module (`neural_network_models.py`) provides a comprehensive ensemble of multiple neural network architectures specifically designed for electromagnetic field (EMF) prediction from transmission line data.

## Features

### 🧠 Multiple Neural Network Architectures (5 Models)

1. **MLP_Standard** - Balanced Architecture
   - **Layers**: 3 hidden layers (100 → 50 → 25)
   - **Activation**: ReLU
   - **Best For**: General-purpose predictions
   - **Regularization**: L2 penalty (α=0.001)

2. **MLP_Deep** - Deep Learning Architecture
   - **Layers**: 5 hidden layers (128 → 64 → 32 → 16 → 8)
   - **Activation**: ReLU
   - **Best For**: Complex non-linear patterns
   - **Regularization**: Light L2 penalty (α=0.0001)

3. **MLP_Wide** - Wide Architecture
   - **Layers**: 3 hidden layers (200 → 150 → 100)
   - **Activation**: ReLU
   - **Best For**: High-dimensional feature interactions
   - **Batch Size**: 64 (larger than others)

4. **MLP_Regularized** - Conservative Model
   - **Layers**: 3 hidden layers (80 → 40 → 20)
   - **Activation**: ReLU
   - **Best For**: Preventing overfitting
   - **Regularization**: Strong L2 penalty (α=0.01)

5. **MLP_Tanh** - Alternative Activation
   - **Layers**: 3 hidden layers (100 → 50 → 25)
   - **Activation**: Tanh (instead of ReLU)
   - **Best For**: Different non-linearity exploration
   - **Output Range**: [-1, 1]

## Key Features

### ✅ Automatic Early Stopping
- All models use **early stopping** with validation monitoring
- Stops training when validation loss doesn't improve for 50 iterations
- Prevents overfitting automatically

### ✅ Adaptive Learning Rate
- Learning rate adjusts during training
- Starts at 0.001 and reduces when needed
- Ensures optimal convergence

### ✅ Validation-Based Model Selection
- Uses separate validation set (15% of training data)
- Best model selected based on validation performance
- Prevents overfitting to training data

### ✅ Comprehensive Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Mean Absolute Percentage Error (MAPE)
- Explained Variance
- Validation R² and RMSE

## Usage

### Basic Usage in Notebook

```python
# Import the module
from neural_network_models import train_neural_network_ensemble

# Train ensemble for E-field
ensemble_E, results_E, best_name_E, best_model_E = train_neural_network_ensemble(
    X_train_E, y_train_E,    # Training data
    X_val_E, y_val_E,        # Validation data
    X_test_E, y_test_E,      # Test data
    field_type='E-field',
    random_state=42
)

# View results
print(results_E)
```

### Advanced Usage

```python
from neural_network_models import NeuralNetworkEnsemble

# Create ensemble
ensemble = NeuralNetworkEnsemble(random_state=42, verbose=True)

# Build models
ensemble.build_models()

# Train all models
ensemble.train_all(X_train, y_train, model_type='E-field')

# Evaluate on test set
results = ensemble.evaluate_all(X_test, y_test, X_val, y_val)

# Get best model
best_name, best_model = ensemble.get_best_model(results)

# Get ensemble prediction (average of all models)
ensemble_pred = ensemble.get_ensemble_prediction(X_test, method='mean')

# Print summary
ensemble.print_summary(results)
```

## File Structure

```
machine learning emf/
├── neural_network_models.py          # Main neural network module
├── EMF_Model_Training_Testing.ipynb  # Main notebook (imports this module)
├── EMF_Data_StandardScaler.csv       # Training data
└── NEURAL_NETWORK_README.md          # This file
```

## Model Comparison

The module automatically:
1. **Trains** all 5 neural network architectures
2. **Evaluates** each on validation and test sets
3. **Ranks** models by R² score
4. **Selects** the best performing architecture
5. **Returns** detailed metrics for all models

## Output Metrics

For each neural network, you'll get:

| Metric | Description |
|--------|-------------|
| MSE | Mean Squared Error (lower is better) |
| RMSE | Root Mean Squared Error (lower is better) |
| MAE | Mean Absolute Error (lower is better) |
| R2_Score | R² Score (higher is better, max 1.0) |
| MAPE (%) | Mean Absolute Percentage Error |
| Explained_Variance | Variance explained by model |
| Iterations | Number of training iterations |
| Val_R2 | Validation R² Score |
| Val_RMSE | Validation RMSE |

## Best Practices

1. **Always use validation set** - Prevents overfitting
2. **Check all architectures** - Different architectures work better for different data
3. **Monitor training iterations** - Models that stop early may need adjustment
4. **Compare with other models** - Neural networks aren't always the best choice
5. **Use ensemble prediction** - Averaging multiple NNs often improves accuracy

## Advantages Over Single Neural Network

✅ **Robustness**: Multiple architectures reduce risk of poor performance  
✅ **Automatic Selection**: Best model chosen automatically  
✅ **Ensemble Options**: Can average predictions from all models  
✅ **Architecture Exploration**: Tests different network designs  
✅ **Better Generalization**: Different models capture different patterns  

## Integration with Main Models

The neural network ensemble integrates seamlessly with:
- Random Forest
- XGBoost
- LightGBM
- Support Vector Regression (SVR)

The best neural network architecture is automatically compared with tree-based and kernel methods to find the overall best model for EMF prediction.

## Performance Tips

1. **Deep vs Wide**:
   - Use **MLP_Deep** for complex non-linear relationships
   - Use **MLP_Wide** for high-dimensional feature interactions

2. **Overfitting Prevention**:
   - Use **MLP_Regularized** if validation R² << training R²
   - Check early stopping iterations

3. **Alternative Activations**:
   - Try **MLP_Tanh** if ReLU models plateau
   - Tanh can work better for normalized data

4. **Ensemble Prediction**:
   - Use `ensemble.get_ensemble_prediction()` for most robust predictions
   - Averages all 5 models for stability

## Troubleshooting

**Problem**: All models stop at max iterations (1000)  
**Solution**: Increase `max_iter` or reduce `learning_rate_init`

**Problem**: Models converge too early (<100 iterations)  
**Solution**: Reduce `learning_rate_init` or adjust `n_iter_no_change`

**Problem**: High training R² but low validation R²  
**Solution**: Use `MLP_Regularized` or increase `alpha` parameter

**Problem**: Very different results between architectures  
**Solution**: Normal - use ensemble prediction for stability

## Future Enhancements

Potential additions to the module:
- CNN-based architectures for spatial patterns
- LSTM for time-series EMF data
- Attention mechanisms
- Hyperparameter tuning with GridSearchCV
- Custom loss functions for EMF-specific constraints

## Citation

If you use this module in your research, please cite:

```
EMF Neural Network Ensemble Module
Author: EMF Prediction Team
Date: October 31, 2025
Application: Electromagnetic Field Prediction from Transmission Lines
```

## License

This module is part of the EMF Prediction project and follows the same license as the main project.

---

**Note**: This module requires the main data to be pre-normalized using StandardScaler. All input data must be normalized before using these neural networks.

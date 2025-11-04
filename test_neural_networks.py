"""
Test Script for Neural Network Ensemble Module
==============================================

This script tests the neural_network_models.py module to ensure
all functions work correctly.

Run this script to verify the module is properly installed and working.
"""

import numpy as np
import sys

print("=" * 70)
print("NEURAL NETWORK ENSEMBLE MODULE - TEST SCRIPT")
print("=" * 70)

# Test 1: Import module
print("\n[Test 1/5] Testing module import...")
try:
    from neural_network_models import NeuralNetworkEnsemble, train_neural_network_ensemble
    print("✓ Module imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create synthetic test data
print("\n[Test 2/5] Creating synthetic test data...")
try:
    np.random.seed(42)
    
    # Create synthetic EMF-like data
    n_samples = 500
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 0.5 + 
         np.random.randn(n_samples) * 0.1)  # Linear + noise
    
    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f"✓ Created synthetic data:")
    print(f"  Training:   {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test:       {X_test.shape[0]} samples")
except Exception as e:
    print(f"✗ Data creation failed: {e}")
    sys.exit(1)

# Test 3: Create ensemble object
print("\n[Test 3/5] Creating NeuralNetworkEnsemble object...")
try:
    ensemble = NeuralNetworkEnsemble(random_state=42, verbose=False)
    print("✓ Ensemble object created")
except Exception as e:
    print(f"✗ Ensemble creation failed: {e}")
    sys.exit(1)

# Test 4: Build models
print("\n[Test 4/5] Building neural network models...")
try:
    ensemble.build_models()
    print(f"✓ Built {len(ensemble.models)} neural network models")
    for name in ensemble.models.keys():
        print(f"  • {name}")
except Exception as e:
    print(f"✗ Model building failed: {e}")
    sys.exit(1)

# Test 5: Train and evaluate
print("\n[Test 5/5] Training and evaluating models...")
try:
    # Train
    ensemble.train_all(X_train, y_train, model_type='Test')
    
    # Evaluate
    results = ensemble.evaluate_all(X_test, y_test, X_val, y_val)
    
    print("✓ All models trained and evaluated successfully")
    print("\nResults Summary:")
    print("-" * 70)
    print(results[['Model', 'R2_Score', 'RMSE', 'Val_R2']].to_string(index=False))
    
    # Get best model
    best_name, best_model = ensemble.get_best_model(results)
    print("\n" + "=" * 70)
    print(f"🏆 Best Model: {best_name}")
    print(f"   Test R²: {results.iloc[0]['R2_Score']:.6f}")
    print(f"   Val R²:  {results.iloc[0]['Val_R2']:.6f}")
    print("=" * 70)
    
except Exception as e:
    print(f"✗ Training/evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Ensemble prediction
print("\n[Test 6/6] Testing ensemble prediction...")
try:
    # Get individual predictions
    predictions = ensemble.predict_all(X_test)
    print(f"✓ Generated predictions from {len(predictions)} models")
    
    # Get ensemble prediction
    ensemble_pred = ensemble.get_ensemble_prediction(X_test, method='mean')
    print(f"✓ Generated ensemble prediction (average of all models)")
    print(f"  Ensemble prediction shape: {ensemble_pred.shape}")
    
except Exception as e:
    print(f"✗ Prediction failed: {e}")
    sys.exit(1)

# Test 7: Quick function test
print("\n[Test 7/7] Testing train_neural_network_ensemble() function...")
try:
    ensemble2, results2, best_name2, best_model2 = train_neural_network_ensemble(
        X_train, y_train, X_val, y_val, X_test, y_test,
        field_type='Quick Test', random_state=42
    )
    print(f"✓ Quick training function works correctly")
    print(f"  Best model: {best_name2}")
    print(f"  Test R²: {results2.iloc[0]['R2_Score']:.6f}")
    
except Exception as e:
    print(f"✗ Quick function failed: {e}")
    sys.exit(1)

# Final summary
print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
print("\nModule Status: READY FOR USE")
print("\nNext Steps:")
print("  1. Import module in your notebook:")
print("     from neural_network_models import train_neural_network_ensemble")
print("  2. Train on your EMF data")
print("  3. Compare results with other models")
print("\nFor help, see:")
print("  • NEURAL_NETWORK_README.md - Full documentation")
print("  • nn_quick_reference.py - Quick reference guide")
print("=" * 70)

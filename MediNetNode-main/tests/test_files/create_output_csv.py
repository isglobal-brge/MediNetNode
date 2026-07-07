"""
Script to generate test CSV files for OUTPUT schema detection.
Creates CSV files representing different types of model outputs.
"""
import pandas as pd
import numpy as np

# 1. Classification Output (single column with class labels)
print("Creating classification output CSV...")
np.random.seed(42)
n_samples = 100

classification_data = {
    'risk_level': np.random.choice(['no_risk', 'at_risk', 'high_risk'], n_samples, p=[0.5, 0.3, 0.2])
}

classification_df = pd.DataFrame(classification_data)
classification_df.to_csv('csv/output_classification.csv', index=False)
print(f"[OK] Created: csv/output_classification.csv (Classification: 3 classes)")

# 2. Binary Classification Output
print("\nCreating binary classification output CSV...")
binary_data = {
    'diagnosis': np.random.choice(['negative', 'positive'], n_samples, p=[0.7, 0.3])
}

binary_df = pd.DataFrame(binary_data)
binary_df.to_csv('csv/output_binary.csv', index=False)
print(f"[OK] Created: csv/output_binary.csv (Binary Classification: 2 classes)")

# 3. Regression Output (single numeric column)
print("\nCreating regression output CSV...")
regression_data = {
    'predicted_glucose_level': np.random.uniform(70, 200, n_samples)
}

regression_df = pd.DataFrame(regression_data)
regression_df.to_csv('csv/output_regression.csv', index=False)
print(f"[OK] Created: csv/output_regression.csv (Regression: continuous values)")

# 4. Multi-Class Probabilities Output (multiple columns with probabilities)
print("\nCreating multi-class probabilities output CSV...")
# Generate probabilities that sum to 1
probs = np.random.dirichlet(np.ones(3), n_samples)

multiclass_prob_data = {
    'prob_no_risk': probs[:, 0],
    'prob_at_risk': probs[:, 1],
    'prob_high_risk': probs[:, 2]
}

multiclass_prob_df = pd.DataFrame(multiclass_prob_data)
multiclass_prob_df.to_csv('csv/output_multiclass_probs.csv', index=False)
print(f"[OK] Created: csv/output_multiclass_probs.csv (Multi-class probabilities)")

# 5. Multi-Output Regression (multiple numeric predictions)
print("\nCreating multi-output regression CSV...")
multioutput_data = {
    'systolic_bp': np.random.uniform(90, 180, n_samples),
    'diastolic_bp': np.random.uniform(60, 120, n_samples),
    'heart_rate': np.random.uniform(60, 100, n_samples)
}

multioutput_df = pd.DataFrame(multioutput_data)
multioutput_df.to_csv('csv/output_multioutput.csv', index=False)
print(f"[OK] Created: csv/output_multioutput.csv (Multi-output regression: 3 values)")

print("\n[OK] All output CSV files created successfully!")
print("\nSummary:")
print("  - output_classification.csv: Classification (3 classes)")
print("  - output_binary.csv: Binary classification (2 classes)")
print("  - output_regression.csv: Regression (continuous)")
print("  - output_multiclass_probs.csv: Multi-class probabilities")
print("  - output_multioutput.csv: Multi-output regression (3 outputs)")

"""
Script to generate test CSV files for model upload testing.
Creates CSV files with different numbers of features for testing.
"""
import pandas as pd
import numpy as np

# 1. Small CSV (10 features)
print("Creating small CSV (10 features)...")
np.random.seed(42)
n_samples = 100

small_data = {
    'age': np.random.randint(18, 90, n_samples),
    'blood_pressure': np.random.uniform(90, 180, n_samples),
    'cholesterol': np.random.uniform(150, 300, n_samples),
    'heart_rate': np.random.randint(60, 100, n_samples),
    'glucose': np.random.uniform(70, 200, n_samples),
    'bmi': np.random.uniform(18.5, 35, n_samples),
    'smoking': np.random.choice([0, 1], n_samples),
    'exercise_hours': np.random.uniform(0, 10, n_samples),
    'stress_level': np.random.randint(1, 11, n_samples),
    'family_history': np.random.choice([0, 1], n_samples),
}

small_df = pd.DataFrame(small_data)
small_df.to_csv('csv/cardio_small.csv', index=False)
print(f"[OK] Created: csv/cardio_small.csv ({len(small_df.columns)} features, {len(small_df)} rows)")

# 2. Medium CSV (100 features)
print("\nCreating medium CSV (100 features)...")
medium_data = {}

# Clinical features (30)
for i in range(30):
    medium_data[f'clinical_{i:02d}'] = np.random.uniform(0, 100, n_samples)

# Lab results (30)
for i in range(30):
    medium_data[f'lab_{i:02d}'] = np.random.uniform(50, 150, n_samples)

# Imaging features (20)
for i in range(20):
    medium_data[f'imaging_{i:02d}'] = np.random.uniform(0, 1, n_samples)

# Patient history (10)
for i in range(10):
    medium_data[f'history_{i:02d}'] = np.random.choice([0, 1], n_samples)

# Medication data (10)
for i in range(10):
    medium_data[f'medication_{i:02d}'] = np.random.choice([0, 1], n_samples)

medium_df = pd.DataFrame(medium_data)
medium_df.to_csv('csv/clinical_medium.csv', index=False)
print(f"[OK] Created: csv/clinical_medium.csv ({len(medium_df.columns)} features, {len(medium_df)} rows)")

# 3. Large CSV (5000 features - genetics)
print("\nCreating large CSV (5000 features)...")
genetics_data = {}

# Gene expression data
for i in range(5000):
    genetics_data[f'gene_{i:04d}'] = np.random.uniform(0, 20, n_samples)

genetics_df = pd.DataFrame(genetics_data)
genetics_df.to_csv('csv/genetics_large.csv', index=False)
print(f"[OK] Created: csv/genetics_large.csv ({len(genetics_df.columns)} features, {len(genetics_df)} rows)")

# 4. Unordered CSV (same as small but columns shuffled)
print("\nCreating unordered CSV (columns shuffled)...")
unordered_df = small_df.copy()
cols = list(unordered_df.columns)
np.random.shuffle(cols)
unordered_df = unordered_df[cols]
unordered_df.to_csv('csv/cardio_unordered.csv', index=False)
print(f"[OK] Created: csv/cardio_unordered.csv ({len(unordered_df.columns)} features, {len(unordered_df)} rows)")
print(f"    Original order: {list(small_df.columns[:5])}...")
print(f"    Shuffled order: {list(unordered_df.columns[:5])}...")

print("\n[OK] All CSV files created successfully!")
print("\nSummary:")
print("  - cardio_small.csv: 10 features, 100 rows")
print("  - clinical_medium.csv: 100 features, 100 rows")
print("  - genetics_large.csv: 5000 features, 100 rows")
print("  - cardio_unordered.csv: 10 features (shuffled), 100 rows")

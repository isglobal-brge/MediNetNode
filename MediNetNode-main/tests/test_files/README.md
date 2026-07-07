# Test Files for Model Upload and Inference

This directory contains test files for validating the model upload and inference functionality.

## Directory Structure

```
test_files/
├── onnx/               # ONNX model files
│   ├── cardionet_small.onnx       (13 KB, 10 features)
│   ├── clinical_medium.onnx       (264 KB, 100 features)
│   └── genetics_large.onnx        (23 MB, 5000 features)
├── csv/                # CSV sample data files
│   ├── cardio_small.csv           (11 KB, 10 features)
│   ├── clinical_medium.csv        (150 KB, 100 features)
│   ├── genetics_large.csv         (8.9 MB, 5000 features)
│   └── cardio_unordered.csv       (11 KB, 10 features shuffled)
└── README.md
```

## ONNX Models

### 1. `cardionet_small.onnx` (10 features)
- **Size**: 13 KB
- **Input Shape**: (batch_size, 10)
- **Output Shape**: (batch_size, 3) - Classification with 3 classes
- **Classes**: no_risk, at_risk, high_risk
- **Architecture**: Simple feedforward neural network (64 → 32 → 3)
- **Use Case**: Testing basic model upload with few features

### 2. `clinical_medium.onnx` (100 features)
- **Size**: 264 KB
- **Input Shape**: (batch_size, 100)
- **Output Shape**: (batch_size, 2) - Binary classification
- **Architecture**: Deeper network with dropout (256 → 128 → 64 → 2)
- **Use Case**: Testing model upload with moderate number of features

### 3. `genetics_large.onnx` (5000 features)
- **Size**: 23 MB
- **Input Shape**: (batch_size, 5000)
- **Output Shape**: (batch_size, 4) - Multi-class classification
- **Architecture**: Large network with dropout (1024 → 512 → 256 → 4)
- **Use Case**: Testing model upload with thousands of features (genetics scenario)

## CSV Sample Files

### 1. `cardio_small.csv` (10 features, 100 rows)
**Features**:
- `age` (integer): Patient age (18-90 years)
- `blood_pressure` (float): Systolic blood pressure (90-180 mmHg)
- `cholesterol` (float): Cholesterol level (150-300 mg/dL)
- `heart_rate` (integer): Heart rate (60-100 bpm)
- `glucose` (float): Blood glucose (70-200 mg/dL)
- `bmi` (float): Body Mass Index (18.5-35)
- `smoking` (binary): Smoking status (0/1)
- `exercise_hours` (float): Weekly exercise hours (0-10)
- `stress_level` (integer): Stress level (1-10)
- `family_history` (binary): Family history of disease (0/1)

**Use Case**: Testing CSV auto-detection with small number of features

### 2. `clinical_medium.csv` (100 features, 100 rows)
**Feature Groups**:
- `clinical_00` to `clinical_29` (30 features): Clinical measurements (0-100)
- `lab_00` to `lab_29` (30 features): Lab test results (50-150)
- `imaging_00` to `imaging_19` (20 features): Imaging features (0-1)
- `history_00` to `history_09` (10 features): Patient history (binary 0/1)
- `medication_00` to `medication_09` (10 features): Medication data (binary 0/1)

**Use Case**: Testing CSV auto-detection with moderate number of features

### 3. `genetics_large.csv` (5000 features, 100 rows)
**Features**:
- `gene_0000` to `gene_4999` (5000 features): Gene expression data (0-20)

**Use Case**: Testing CSV auto-detection with thousands of features (genetics scenario)

### 4. `cardio_unordered.csv` (10 features shuffled, 100 rows)
**Same features as `cardio_small.csv` but columns in different order**:
- Original order: `age, blood_pressure, cholesterol, heart_rate, glucose, bmi, smoking, exercise_hours, stress_level, family_history`
- Shuffled order: `age, cholesterol, bmi, exercise_hours, family_history, blood_pressure, heart_rate, smoking, glucose, stress_level`

**Use Case**: Testing inference with CSV columns in wrong order (must be reordered to match model input schema)

## Testing Scenarios

### Scenario 1: Upload Small Model
1. Navigate to: http://localhost:5001/inference/models/upload/
2. Upload `cardionet_small.onnx`
3. Upload `cardio_small.csv` for input schema auto-detection
4. Verify 10 features detected correctly with proper types

### Scenario 2: Upload Medium Model
1. Upload `clinical_medium.onnx`
2. Upload `clinical_medium.csv` for input schema auto-detection
3. Verify 100 features detected correctly

### Scenario 3: Upload Large Genetics Model
1. Upload `genetics_large.onnx`
2. Upload `genetics_large.csv` for input schema auto-detection
3. Verify 5000 features detected correctly
4. Verify UI handles large number of features efficiently

### Scenario 4: Test Column Reordering During Inference
1. Upload model with `cardio_small.csv` (original order)
2. Perform inference with `cardio_unordered.csv` (shuffled order)
3. Verify system correctly reorders columns to match model input schema
4. Verify inference results are valid

## CSV Output Files (for Output Schema Auto-detection)

### 1. `output_classification.csv` (Classification - 3 classes)
**Features**:
- `risk_level` (categorical): Values are `no_risk`, `at_risk`, `high_risk`

**Use Case**: Testing output schema auto-detection for multi-class classification

### 2. `output_binary.csv` (Binary Classification)
**Features**:
- `diagnosis` (categorical): Values are `negative`, `positive`

**Use Case**: Testing output schema auto-detection for binary classification

### 3. `output_regression.csv` (Regression)
**Features**:
- `predicted_glucose_level` (float): Continuous values (70-200)

**Use Case**: Testing output schema auto-detection for regression models

### 4. `output_multiclass_probs.csv` (Multi-Class Probabilities)
**Features**:
- `prob_no_risk` (float): Probability values (0-1)
- `prob_at_risk` (float): Probability values (0-1)
- `prob_high_risk` (float): Probability values (0-1)

**Note**: Values sum to 1.0 for each row (proper probability distribution)

**Use Case**: Testing output schema auto-detection for models that output class probabilities

### 5. `output_multioutput.csv` (Multi-Output Regression)
**Features**:
- `systolic_bp` (float): Systolic blood pressure (90-180)
- `diastolic_bp` (float): Diastolic blood pressure (60-120)
- `heart_rate` (float): Heart rate (60-100)

**Use Case**: Testing output schema auto-detection for models with multiple regression outputs

## Regenerating Test Files

If you need to regenerate the test files:

```bash
cd tests/test_files
python create_test_onnx.py      # Regenerate ONNX models
python create_test_csv.py       # Regenerate input CSV files
python create_output_csv.py     # Regenerate output CSV files
```

## Notes

- All CSV files contain 100 sample rows for testing
- All numeric values are randomly generated within realistic ranges
- ONNX models are trained with random weights (for testing upload only, not for real inference)
- File sizes are realistic for their respective feature counts

import torch
import pandas as pd
import os
import json
import numpy as np
from typing import Tuple, Optional, Union
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from datasets.utils.logging import disable_progress_bar
from sklearn.preprocessing import LabelEncoder, StandardScaler


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
disable_progress_bar()

def load_data_from_django(dataset_id: int, use_experiment: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load data from Django dataset system.

    Args:
        dataset_id: ID of dataset in Django system
        use_experiment: If True, use experiment_file_path instead of file_path

    Returns:
        Tuple of (data_df, target_column) or (None, None) if error
    """
    try:
        # Import Django models (must be done here to avoid import issues)
        import os
        import sys
        import django

        # Setup Django environment (safe initialization)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if project_root not in sys.path:
            sys.path.append(project_root)

        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medinet.settings')

        # Safe Django setup - avoid multiple initialization
        try:
            if not django.apps.apps.ready:
                django.setup()
        except RuntimeError as e:
            if "Django is already configured" not in str(e):
                raise

        from dataset.models import Dataset

        print(f"[SEARCH] Fetching dataset {dataset_id} from Django...")

        # Get dataset from Django
        try:
            dataset = Dataset.objects.using('datasets_db').get(id=dataset_id)
            print(f"[OK] Dataset found: {dataset.name}")
        except Dataset.DoesNotExist:
            print(f"[ERROR] Dataset {dataset_id} not found in Django system")
            return None, None

        # Resolve which file path to use
        file_path = dataset.file_path
        if use_experiment and dataset.experiment_file_path and os.path.exists(dataset.experiment_file_path):
            file_path = dataset.experiment_file_path
            print(f"[EXP] Using experimental subset: {file_path}")
        elif use_experiment:
            print(f"[EXP] use_experiment=True but no experiment file available — falling back to production file")

        # Get file path
        if not file_path or not os.path.exists(file_path):
            print(f"[ERROR] Dataset file not found: {file_path}")
            return None, None

        print(f"📂 Loading data from: {file_path}")

        # Load CSV data
        data_df = pd.read_csv(file_path)
        
        if data_df.empty:
            print(f"[ERROR] Dataset file is empty: {file_path}")
            return None, None
            
        print(f"[OK] Data loaded: {len(data_df)} rows, {len(data_df.columns)} columns")
        
        # Get target column from dataset metadata
        target_column = None
        if dataset.target_column:
            try:
                target_column = dataset.target_column
                print(f"[INIT] Target column from metadata: {target_column}")
            except json.JSONDecodeError:
                print("[WARNING] Warning: Could not parse target_column JSON")
        
        # Fallback: try common target column names
        if not target_column or target_column not in data_df.columns:
            common_targets = ['target', 'label', 'class', 'y', 'output', 'result']
            for col in common_targets:
                if col in data_df.columns:
                    target_column = col
                    print(f"[SYNC] Using fallback target column: {target_column}")
                    break
        
        # Final fallback: use last column
        if not target_column or target_column not in data_df.columns:
            target_column = data_df.columns[-1]
            print(f"[WARNING] Using last column as target: {target_column}")
            
        return data_df, target_column
        
    except Exception as e:
        print(f"[ERROR] Error loading dataset {dataset_id}: {str(e)}")
        import traceback
        print(f"[LIST] Full traceback: {traceback.format_exc()}")
        return None, None

class SQLiteDataset(Dataset):
    """PyTorch Dataset for pre-loaded features and targets."""
    
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        """
        Initialize a dataset with pre-loaded features and targets
        
        Args:
            features: Feature tensor (X)
            targets: Target tensor (y)
        """
        self.X = features
        self.y = targets
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

def prepare_dataset(
    data_df: pd.DataFrame,
    target_column: str,
    scaler: Optional[StandardScaler] = None,
    label_encoder: Optional[LabelEncoder] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare dataset from a pandas DataFrame.

    Args:
        data_df: DataFrame with all data.
        target_column: Name of the target column.
        scaler: A *fitted* StandardScaler to apply to features.  When provided
            the scaler is applied with ``transform()`` (not ``fit_transform``).
            Pass ``None`` to skip normalisation (not recommended for clinical data).
        label_encoder: A *fitted* LabelEncoder for string targets.  When provided
            ``transform()`` is used instead of ``fit_transform`` so that the same
            mapping is applied consistently across train and val splits.

    Returns:
        Tuple of (features_tensor, targets_tensor)
    """
    X_raw = np.vstack(data_df.drop(columns=[target_column]).values).astype(np.float32)

    if scaler is not None:
        X_raw = scaler.transform(X_raw).astype(np.float32)

    X_tensor = torch.from_numpy(X_raw)

    if pd.api.types.is_string_dtype(data_df[target_column]):
        if label_encoder is not None:
            y_encoded = label_encoder.transform(data_df[target_column].values)
        else:
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(data_df[target_column].values)
        y_tensor = torch.from_numpy(y_encoded).long()
    else:
        y_tensor = torch.from_numpy(data_df[target_column].values.astype(np.float32))

    return X_tensor, y_tensor


def load_ml_data(
    dataset_id: int,
    val_size: float = 0.2,
    random_state: int = 42,
    use_experiment: bool = False,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load data as NumPy arrays for ML algorithms (SVM, Random Forest, etc.).

    Unlike create_train_val_loaders(), this function returns raw NumPy arrays
    instead of PyTorch DataLoaders, since traditional ML algorithms don't use
    mini-batch training.

    Args:
        dataset_id (int): ID of dataset in Django system
        val_size (float): Size of validation set (proportion, 0.0-1.0)
        random_state (int): Random seed for reproducibility

    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val)) where all are numpy arrays

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If dataset loading fails

    Example:
        >>> (X_train, y_train), (X_val, y_val) = load_ml_data(dataset_id=1)
        >>> print(X_train.shape, y_train.shape)
        (800, 30) (800,)
        >>> print(X_val.shape, y_val.shape)
        (200, 30) (200,)
    """
    # Input validation
    if not isinstance(dataset_id, int) or dataset_id <= 0:
        raise ValueError(f"dataset_id must be positive integer, got: {dataset_id}")
    if not 0.0 <= val_size <= 1.0:
        raise ValueError(f"val_size must be between 0.0 and 1.0, got: {val_size}")

    print(f"\n{'='*60}")
    print(f"[PACKAGE] Loading ML Data (NumPy Arrays)")
    print(f"{'='*60}")
    print(f"   Dataset ID: {dataset_id}")
    print(f"   Validation split: {val_size:.1%}")
    print(f"   Random state: {random_state}")

    try:
        # Load dataset from Django system
        data_df, target_column = load_data_from_django(dataset_id, use_experiment=use_experiment)

        if data_df is None or target_column is None:
            error_msg = f"Failed to load dataset {dataset_id} from Django system"
            print(f"[ERROR] {error_msg}")
            raise RuntimeError(error_msg)

        print(f"[OK] Dataset loaded successfully")
        print(f"   Total rows: {len(data_df)}")
        print(f"   Columns: {len(data_df.columns)}")
        print(f"   Target column: {target_column}")

        # Split data into train and validation sets
        print(f"\n🔀 Splitting data...")
        train_df, val_df = train_test_split(
            data_df,
            test_size=val_size,
            random_state=random_state,
            stratify=data_df[target_column] if len(data_df[target_column].unique()) > 1 else None
        )

        print(f"   Train set: {len(train_df)} samples ({(1-val_size)*100:.1f}%)")
        print(f"   Val set:   {len(val_df)} samples ({val_size*100:.1f}%)")

        # Convert to NumPy arrays (NO PyTorch tensors for ML)
        print(f"\n[CONFIG] Converting to NumPy arrays...")

        feature_cols = [c for c in data_df.columns if c != target_column]

        # Training data
        X_train = train_df[feature_cols].values.astype(np.float64)
        y_train = train_df[target_column].values

        # Handle categorical target (encode to integers)
        le: Optional[LabelEncoder] = None
        if pd.api.types.is_string_dtype(y_train) or pd.api.types.is_categorical_dtype(y_train):
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            print(f"   Encoded target labels: {dict(enumerate(le.classes_))}")
        else:
            y_train = y_train.astype(np.int64)

        # Fit StandardScaler on training features, apply to val.
        ml_scaler = StandardScaler()
        X_train = ml_scaler.fit_transform(X_train)
        print(f"   StandardScaler fitted on {len(train_df)} training samples")

        # Validation data
        X_val = val_df[feature_cols].values.astype(np.float64)
        X_val = ml_scaler.transform(X_val)
        y_val = val_df[target_column].values

        if pd.api.types.is_string_dtype(y_val) or pd.api.types.is_categorical_dtype(y_val):
            if le is not None:
                y_val = le.transform(y_val)
            else:
                le = LabelEncoder()
                y_val = le.fit_transform(y_val)
        else:
            y_val = y_val.astype(np.int64)

        print(f"[OK] NumPy arrays created successfully")
        print(f"\n[INFO] Data Summary:")
        print(f"   X_train shape: {X_train.shape} (samples, features)")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   X_val shape:   {X_val.shape}")
        print(f"   y_val shape:   {y_val.shape}")
        print(f"   Data type:     {X_train.dtype}")
        print(f"   Target type:   {y_train.dtype}")
        print(f"   Unique labels: {np.unique(y_train)}")
        print(f"{'='*60}\n")

        return (X_train, y_train), (X_val, y_val)

    except (ValueError, RuntimeError):
        # Re-raise validation and runtime errors with context preserved
        raise
    except Exception as e:
        error_msg = f"Unexpected error loading ML data for dataset {dataset_id}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        print(f"[LIST] Full traceback: {traceback.format_exc()}")
        raise RuntimeError(error_msg) from e


def create_train_val_loaders(
    dataset_id: int,
    val_size: float = 0.2,
    batch_size: int = 32,
    random_state: int = 42,
    use_experiment: bool = False,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """
    Create train and validation dataloaders from Django dataset system.
    
    Args:
        dataset_id (int): ID of dataset in Django system
        val_size (float): Size of validation set (proportion, 0.0-1.0)
        batch_size (int): Batch size for dataloaders
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, val_loader) or (None, None) if error
        
    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If dataset loading fails
    """
    # Input validation
    if not isinstance(dataset_id, int) or dataset_id <= 0:
        raise ValueError(f"dataset_id must be positive integer, got: {dataset_id}")
    if not 0.0 <= val_size <= 1.0:
        raise ValueError(f"val_size must be between 0.0 and 1.0, got: {val_size}")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be positive integer, got: {batch_size}")
    
    print(f"[SEARCH] Loading dataset {dataset_id} from Django system...")
    
    try:
        # Load dataset from Django system
        data_df, target_column = load_data_from_django(dataset_id, use_experiment=use_experiment)

        if data_df is None or target_column is None:
            error_msg = f"Failed to load dataset {dataset_id} from Django system"
            print(f"[ERROR] {error_msg}")
            raise RuntimeError(error_msg)

        print(f"[OK] Dataset loaded: {len(data_df)} rows, target column: {target_column}")
        
        # Split data into train and validation sets — use stratify to preserve
        # class balance across splits (critical for imbalanced clinical datasets).
        print(f"🔀 Splitting data: {1-val_size:.1%} train, {val_size:.1%} validation...")
        try:
            stratify_col = data_df[target_column] if data_df[target_column].nunique() > 1 else None
            train_df, val_df = train_test_split(
                data_df, test_size=val_size, random_state=random_state, stratify=stratify_col
            )
        except ValueError:
            # Stratification can fail for very small datasets — fall back gracefully
            print("[WARNING] Stratified split failed — falling back to random split")
            train_df, val_df = train_test_split(data_df, test_size=val_size, random_state=random_state)

        print(f"[INFO] Train set: {len(train_df)} samples, Val set: {len(val_df)} samples")

        # Fit StandardScaler on train features only, then apply to val.
        # Clinical tabular data (age, lab values, vitals) has wildly different
        # scales — without normalisation gradient updates are dominated by the
        # largest-magnitude features regardless of their importance.
        feature_cols = [c for c in data_df.columns if c != target_column]
        scaler = StandardScaler()
        scaler.fit(train_df[feature_cols].values.astype(np.float32))
        print(f"[NORM] StandardScaler fitted on {len(train_df)} training samples ({len(feature_cols)} features)")

        # Build a shared LabelEncoder for string targets so val uses the same mapping
        label_encoder: Optional[LabelEncoder] = None
        if pd.api.types.is_string_dtype(data_df[target_column]):
            label_encoder = LabelEncoder()
            label_encoder.fit(data_df[target_column].values)

        # Prepare tensors
        print("[CONFIG] Converting to PyTorch tensors...")
        X_train, y_train = prepare_dataset(train_df, target_column, scaler=scaler, label_encoder=label_encoder)
        X_val, y_val = prepare_dataset(val_df, target_column, scaler=scaler, label_encoder=label_encoder)
        
        print(f"[INIT] Tensor shapes - Train: X{X_train.shape}, y{y_train.shape} | Val: X{X_val.shape}, y{y_val.shape}")
        
        # Create datasets
        train_dataset = SQLiteDataset(X_train, y_train)
        val_dataset = SQLiteDataset(X_val, y_val)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"🚀 DataLoaders created successfully with batch_size={batch_size}")
        return train_loader, val_loader
        
    except (ValueError, RuntimeError):
        # Re-raise validation and runtime errors with context preserved
        raise
    except Exception as e:
        error_msg = f"Unexpected error creating dataloaders for dataset {dataset_id}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        print(f"[LIST] Full traceback: {traceback.format_exc()}")
        raise RuntimeError(error_msg) from e




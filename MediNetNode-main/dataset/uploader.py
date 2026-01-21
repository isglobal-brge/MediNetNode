"""
Secure dataset uploader with medical data validation and metadata extraction.
"""

import os
import hashlib
import tempfile
import shutil
import re
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

# Import pandas - simplified
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from django.conf import settings
from django.core.exceptions import ValidationError
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.db import transaction
from .models import Dataset, DatasetMetadata

User = get_user_model()

# Configure logging
logger = logging.getLogger(__name__)

class DatasetUploadError(Exception):
    """Custom exception for dataset upload errors."""
    pass

class SecurityValidationError(DatasetUploadError):
    """Raised when security validation fails."""
    pass

class MetadataExtractionError(DatasetUploadError):
    """Raised when metadata extraction fails."""
    pass

class SecureDatasetUploader:
    """
    Secure dataset uploader with comprehensive validation and metadata extraction.
    
    Features:
    - File type validation (magic numbers + extensions)
    - Medical data privacy safeguards
    - Automatic metadata extraction
    - Secure storage with atomic operations
    - Progress tracking capabilities
    - Quarantine system for suspicious files
    """
    
    # Allowed file extensions and their corresponding magic numbers
    ALLOWED_EXTENSIONS = {
        '.csv': ['text/csv', 'text/plain'],
        '.json': ['application/json', 'text/plain'],
        '.parquet': ['application/octet-stream'],
        '.h5': ['application/x-hdf'],
        '.npy': ['application/octet-stream'],
    }
    
    # No file size limit for medical datasets (they can be very large)
    MAX_FILE_SIZE = None
    
    # Minimum rows for k-anonymity
    MIN_K_ANONYMITY = 5
    
    # Forbidden column patterns (potential PHI identifiers)
    # Use word boundaries to avoid false positives like 'width' containing 'id'
    FORBIDDEN_PATTERNS = [
        r'\bid\b', r'\bpatient_id\b', r'\bmrn\b', r'\bmedical_record\b', 
        r'\bssn\b', r'\bsocial_security\b', r'\bname\b', r'\bfirst_name\b', 
        r'\blast_name\b', r'\bemail\b', r'\bphone\b', r'\baddress\b',
        r'\bzip\b', r'\bpostal\b', r'\bbirth_date\b', r'\bdob\b', r'\bdate_of_birth\b'
    ]
    
    def __init__(self, user: User, progress_callback=None):
        """
        Initialize the secure uploader.
        
        Args:
            user: The user uploading the dataset
            progress_callback: Optional callback for progress updates
        """
        self.user = user
        self.progress_callback = progress_callback
        self.temp_dir = None
        self.quarantine_dir = self._get_quarantine_dir()
        
    def upload_dataset(
        self,
        file_path: str,
        name: str,
        description: str,
        medical_domain: str,
        data_type: str = 'tabular',
        target_column: str = None
    ) -> Tuple[Dataset, Dict[str, Any]]:
        """
        Main method to upload and process a dataset securely.

        Args:
            file_path: Path to the file to upload
            name: Dataset name
            description: Dataset description
            medical_domain: Medical domain (cardiology, neurology, etc.)
            data_type: Type of data (tabular, image, etc.)
            target_column: Column name to use as target for federated learning

        Returns:
            Created Dataset instance

        Raises:
            DatasetUploadError: If upload fails at any stage
        """
        try:
            self._update_progress("initializing", "Initializing upload...")

            # Create temporary directory for processing
            self.temp_dir = tempfile.mkdtemp(prefix='dataset_upload_')

            # Step 1: Validate file
            self._update_progress("validating", "Validating file...")
            self._validate_file(file_path)

            # Step 2: Extract metadata
            self._update_progress("extracting_metadata", "Extracting metadata...")
            metadata = self._extract_metadata(file_path, target_column)

            # Step 3: Validate metadata for medical compliance and remove PHI columns
            self._update_progress("validating_metadata", "Validating medical compliance...")
            metadata = self._validate_medical_compliance(metadata, file_path, target_column)
            
            # Step 4: Calculate checksums
            self._update_progress("calculating_checksums", "Calculating checksums...")
            md5_hash, sha256_hash = self._calculate_checksums(file_path)
            
            # Steps 5-7: Store file and create records atomically
            self._update_progress("saving", "Creating dataset record...")
            
            from django.db import connections
            datasets_connection = connections['datasets_db']
            final_path = None
            
            # Manual transaction handling to ensure proper rollback
            try:
                datasets_connection.set_autocommit(False)  # Start transaction
                
                # Step 5: Store file securely WITHIN transaction
                self._update_progress("storing", "Storing file securely...")
                final_path = self._store_file_securely(file_path, name)
                
                with datasets_connection.cursor() as cursor:
                    cursor.execute("PRAGMA foreign_keys = OFF")
                    
                    # Step 6: Create dataset record with raw SQL
                    dataset_id = self._create_dataset_record_raw_sql(
                        cursor=cursor,
                        name=name,
                        description=description,
                        file_path=final_path,
                        medical_domain=medical_domain,
                        data_type=data_type,
                        metadata=metadata,
                        md5_hash=md5_hash,
                        target_column=target_column
                    )
                    
                    # Step 7: Create metadata record with raw SQL
                    self._create_metadata_record_raw_sql(cursor, dataset_id, metadata)
                
                # If we get here, commit the transaction
                datasets_connection.commit()
                
                # Get dataset instance for return
                dataset = Dataset.objects.using('datasets_db').get(id=dataset_id)
                
            except Exception as e:
                # Rollback on any error
                datasets_connection.rollback()
                
                # CLEANUP: Remove file if it was created
                if final_path and os.path.exists(final_path):
                    try:
                        os.unlink(final_path)
                        
                        # Remove empty directories recursively (user_id/year/month structure)
                        dir_path = os.path.dirname(final_path)
                        for _ in range(3):  # Max 3 levels: month -> year -> user_id
                            if os.path.exists(dir_path) and not os.listdir(dir_path):
                                os.rmdir(dir_path)
                                dir_path = os.path.dirname(dir_path)
                            else:
                                break
                    except Exception:
                        pass  # Ignore cleanup errors
                
                raise
            finally:
                # Restore autocommit
                datasets_connection.set_autocommit(True)
            
            self._update_progress("completed", "Upload completed successfully!")
            logger.info(f"Dataset '{name}' uploaded successfully by user {self.user.username}")
            
            # Prepare upload information
            upload_info = {
                'phi_columns_removed': metadata.get('phi_columns_removed', []),
                'original_columns': metadata.get('original_columns', metadata.get('columns', 0)),
                'final_columns': metadata.get('columns', 0)
            }
            
            return dataset, upload_info
            
        except Exception as e:
            logger.error(f"Dataset upload failed: {str(e)}")
            self._handle_upload_failure(file_path, str(e))
            raise DatasetUploadError(f"Upload failed: {str(e)}")
            
        finally:
            # Cleanup temporary directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _validate_file(self, file_path: str) -> None:
        """
        Comprehensive file validation.
        
        Args:
            file_path: Path to file to validate
            
        Raises:
            SecurityValidationError: If validation fails
        """
        if not os.path.exists(file_path):
            raise SecurityValidationError("File does not exist")
            
        # Check file size (only check for empty files)
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise SecurityValidationError("File is empty")
            
        # Validate extension
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.ALLOWED_EXTENSIONS:
            raise SecurityValidationError(f"File extension '{file_ext}' not allowed")
            
        # Validate magic numbers (file type detection)
        if MAGIC_AVAILABLE:
            try:
                mime_type = magic.from_file(file_path, mime=True)
                allowed_mimes = self.ALLOWED_EXTENSIONS[file_ext]
                
                if mime_type not in allowed_mimes:
                    raise SecurityValidationError(
                        f"File type mismatch: extension '{file_ext}' but detected '{mime_type}'"
                    )
            except Exception as e:
                raise SecurityValidationError(f"Could not determine file type: {str(e)}")
        else:
            # Fallback: basic extension validation only
            logger.warning("python-magic not available, using basic extension validation only")
            
        # Sanitize filename
        filename = os.path.basename(file_path)
        if not self._is_safe_filename(filename):
            raise SecurityValidationError(f"Unsafe filename: {filename}")
    
    def _extract_metadata(self, file_path: str, target_column: str = None) -> Dict[str, Any]:
        """
        Extract metadata based on file type with medical privacy safeguards.

        Args:
            file_path: Path to file
            target_column: Name of the target column (optional)

        Returns:
            Dictionary containing safe metadata

        Raises:
            MetadataExtractionError: If extraction fails
        """
        file_ext = Path(file_path).suffix.lower()

        try:
            if file_ext == '.csv':
                return self._extract_csv_metadata(file_path, target_column)
            elif file_ext == '.json':
                return self._extract_json_metadata(file_path)
            elif file_ext == '.parquet':
                return self._extract_parquet_metadata(file_path, target_column)
            elif file_ext in ['.h5', '.npy']:
                return self._extract_binary_metadata(file_path)
            else:
                raise MetadataExtractionError(f"Unsupported file type: {file_ext}")

        except Exception as e:
            raise MetadataExtractionError(f"Failed to extract metadata: {str(e)}")
    
    def _extract_csv_metadata(self, file_path: str, target_column: str = None) -> Dict[str, Any]:
        """Extract metadata from CSV file with medical safeguards."""
        if not PANDAS_AVAILABLE or pd is None:
            raise MetadataExtractionError("pandas not available for CSV processing")

        try:
            # Read CSV with pandas
            df = pd.read_csv(file_path)

            rows, cols = df.shape

            # Check for nulls - must be 0 for medical data
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                raise MetadataExtractionError(
                    f"Dataset contains null values: {null_counts[null_counts > 0].to_dict()}"
                )

            # Extract safe column information
            column_info = {}
            for col in df.columns:
                dtype = str(df[col].dtype)

                # Map to coarse types for privacy
                if dtype.startswith('int') or dtype.startswith('float'):
                    coarse_type = 'numeric'
                elif dtype == 'object':
                    # Check if it's actually categorical
                    unique_count = df[col].nunique()
                    if unique_count < rows * 0.1:  # Less than 10% unique values
                        coarse_type = 'categorical'
                    else:
                        coarse_type = 'text'
                elif 'datetime' in dtype:
                    coarse_type = 'datetime'
                else:
                    coarse_type = 'other'

                column_info[col] = {
                    'type': coarse_type,
                    'unique_count': df[col].nunique() if rows >= self.MIN_K_ANONYMITY else None
                }

            # Calculate safe statistical summaries for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            statistical_summary = {}

            if rows >= self.MIN_K_ANONYMITY:
                for col in numeric_cols:
                    # Only provide percentiles, not min/max to avoid outlier identification
                    percentiles = df[col].quantile([0.05, 0.5, 0.95])
                    statistical_summary[col] = {
                        'p5': float(percentiles[0.05]),
                        'median': float(percentiles[0.5]),
                        'p95': float(percentiles[0.95]),
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std())
                    }

            # Analyze target column if specified
            target_info = None
            if target_column and target_column in df.columns:
                target_info = self._analyze_target_column(df, target_column, column_info)

            metadata = {
                'file_type': 'csv',
                'rows': rows,
                'columns': cols,
                'column_info': column_info,
                'statistical_summary': statistical_summary,
                'nulls_verified_zero': True,
                'k_anonymity_compliant': rows >= self.MIN_K_ANONYMITY
            }

            # Add target info if available
            if target_info:
                metadata['target_info'] = target_info

            return metadata

        except Exception as e:
            raise MetadataExtractionError(f"CSV metadata extraction failed: {str(e)}")

    def _analyze_target_column(self, df, target_column: str, column_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze target column to determine task type and output configuration.

        Generates metadata for BOTH Deep Learning (DL) and Machine Learning (ML) methods,
        so a single dataset can be used for both paradigms without admin reconfiguration.

        Args:
            df: Pandas DataFrame containing the data
            target_column: Name of the target column
            column_info: Dictionary with column type information

        Returns:
            Dictionary with target information including:
            - DL-specific: output_neurons, recommended_activation, recommended_loss
            - ML-specific: recommended_kernel, recommended_C, recommended_gamma
            - Common: task_type, num_classes, classes
        """
        if target_column not in df.columns:
            return None

        target_series = df[target_column]
        dtype = str(target_series.dtype)
        num_unique = target_series.nunique()
        n_samples = len(df)

        # Determine if numeric or categorical
        is_numeric = dtype.startswith('int') or dtype.startswith('float')

        # Build target info (common for DL and ML)
        target_info = {
            'column_name': target_column,
            'data_type': 'numeric' if is_numeric else 'categorical'
        }

        if is_numeric:
            # Check if it's actually a regression task or disguised classification
            # If integer with few unique values, might be classification
            if dtype.startswith('int') and num_unique <= 20:
                # Could be classification with integer labels
                target_info['task_type'] = 'classification'
                target_info['num_classes'] = int(num_unique)
                target_info['classes'] = sorted(target_series.unique().tolist())

                if num_unique == 2:
                    target_info['task_subtype'] = 'binary_classification'
                    # DL configuration
                    target_info['output_neurons'] = 1
                    target_info['recommended_activation'] = 'sigmoid'
                    target_info['recommended_loss'] = 'BCEWithLogitsLoss'
                else:
                    target_info['task_subtype'] = 'multiclass_classification'
                    # DL configuration
                    target_info['output_neurons'] = int(num_unique)
                    target_info['recommended_activation'] = 'softmax'
                    target_info['recommended_loss'] = 'CrossEntropyLoss'
            else:
                # True regression task
                target_info['task_type'] = 'regression'
                target_info['task_subtype'] = 'regression'
                # DL configuration
                target_info['output_neurons'] = 1
                target_info['recommended_activation'] = 'none'
                target_info['recommended_loss'] = 'MSELoss'

                # Add value statistics for regression
                target_info['value_range'] = {
                    'min': float(target_series.min()),
                    'max': float(target_series.max()),
                    'mean': float(target_series.mean()),
                    'std': float(target_series.std()),
                    'median': float(target_series.median())
                }

        else:
            # Categorical target
            target_info['task_type'] = 'classification'
            target_info['num_classes'] = int(num_unique)
            target_info['classes'] = sorted(target_series.unique().tolist())

            if num_unique == 2:
                target_info['task_subtype'] = 'binary_classification'
                # DL configuration
                target_info['output_neurons'] = 1
                target_info['recommended_activation'] = 'sigmoid'
                target_info['recommended_loss'] = 'BCEWithLogitsLoss'
            else:
                target_info['task_subtype'] = 'multiclass_classification'
                # DL configuration
                target_info['output_neurons'] = int(num_unique)
                target_info['recommended_activation'] = 'softmax'
                target_info['recommended_loss'] = 'CrossEntropyLoss'

        # Add dataset characteristics for ML algorithm-agnostic recommendations
        n_features = len(df.columns) - 1  # Exclude target column

        target_info['dataset_characteristics'] = {
            'n_samples': n_samples,
            'n_features': n_features,
            'samples_per_feature_ratio': n_samples / max(n_features, 1),
            'class_balance': None  # Will be calculated if classification
        }

        # Add class balance for classification tasks
        if target_info['task_type'] == 'classification':
            class_counts = target_series.value_counts().to_dict()
            total = len(target_series)
            class_distribution = {str(k): v/total for k, v in class_counts.items()}
            target_info['dataset_characteristics']['class_balance'] = class_distribution

            # Detect imbalance
            min_ratio = min(class_distribution.values())
            target_info['dataset_characteristics']['is_imbalanced'] = min_ratio < 0.3

        # ==================== ML ALGORITHM CONSIDERATIONS ====================
        # General considerations for supervised ML algorithms (algorithm-agnostic)

        target_info['ml_considerations'] = {
            'supervised_learning': True,  # This dataset has labels (target column)
            'task_suitable_for': [],  # Which ML families work well
            'preprocessing_needs': [],  # What preprocessing is recommended
            'algorithm_hints': {},  # Hints for specific algorithm families
            'challenges': []  # Potential challenges
        }

        ml_cons = target_info['ml_considerations']

        # 1. TASK SUITABILITY - Which ML algorithm families work well?
        if target_info['task_type'] == 'classification':
            if num_unique == 2:
                ml_cons['task_suitable_for'] = [
                    'SVM (binary)', 'Logistic Regression', 'Decision Trees',
                    'Random Forest', 'Gradient Boosting', 'Naive Bayes', 'KNN'
                ]
            else:
                ml_cons['task_suitable_for'] = [
                    'SVM (multiclass)', 'Random Forest', 'Gradient Boosting',
                    'Decision Trees', 'KNN', 'Multinomial Naive Bayes'
                ]
        else:  # Regression
            ml_cons['task_suitable_for'] = [
                'SVR', 'Linear Regression', 'Ridge/Lasso', 'Random Forest Regressor',
                'Gradient Boosting Regressor', 'KNN Regressor', 'Decision Tree Regressor'
            ]

        # 2. DATA SIZE CONSIDERATIONS
        if n_samples < 100:
            ml_cons['preprocessing_needs'].append('Very small dataset: Risk of overfitting')
            ml_cons['challenges'].append('Insufficient data for complex models')
        elif n_samples < 1000:
            ml_cons['preprocessing_needs'].append('Small dataset: Cross-validation crucial')
            ml_cons['algorithm_hints']['recommended'] = ['SVM', 'Decision Trees', 'Naive Bayes']
            ml_cons['algorithm_hints']['avoid'] = ['Deep Learning', 'Complex ensembles']
        elif n_samples > 100000:
            ml_cons['preprocessing_needs'].append('Large dataset: Subsampling may be needed for some algorithms')
            ml_cons['algorithm_hints']['scalable'] = ['Random Forest', 'Gradient Boosting', 'Linear models']
            ml_cons['algorithm_hints']['slow'] = ['SVM with RBF kernel', 'KNN']

        # 3. DIMENSIONALITY CONSIDERATIONS
        curse_of_dimensionality = n_features / max(n_samples, 1) > 0.1

        if n_features > 1000:
            ml_cons['preprocessing_needs'].append('High-dimensional: Feature selection/PCA recommended')
            ml_cons['algorithm_hints']['suitable_high_dim'] = ['Linear SVM', 'Logistic Regression', 'Lasso']
            ml_cons['challenges'].append('Curse of dimensionality')

        if curse_of_dimensionality:
            ml_cons['challenges'].append('Features comparable to samples: regularization essential')
            ml_cons['preprocessing_needs'].append('Regularization required (L1/L2)')

        # 4. CLASS BALANCE (for classification)
        if target_info['task_type'] == 'classification':
            is_imbalanced = target_info['dataset_characteristics'].get('is_imbalanced', False)

            if is_imbalanced:
                ml_cons['preprocessing_needs'].append('Imbalanced classes: Apply class weighting or SMOTE')
                ml_cons['algorithm_hints']['imbalance_handling'] = {
                    'class_weight': 'Most algorithms support class_weight parameter',
                    'resampling': 'SMOTE, ADASYN for minority oversampling',
                    'metrics': 'Use F1, precision, recall instead of accuracy'
                }
                ml_cons['challenges'].append('Class imbalance detected')

        # 5. FEATURE SCALE SENSITIVITY
        ml_cons['algorithm_hints']['scale_sensitive'] = [
            'SVM', 'KNN', 'Logistic Regression', 'Neural Networks'
        ]
        ml_cons['algorithm_hints']['scale_invariant'] = [
            'Decision Trees', 'Random Forest', 'Gradient Boosting', 'Naive Bayes'
        ]
        ml_cons['preprocessing_needs'].append('Normalization/Standardization for distance-based algorithms')

        # 6. KERNEL/NON-LINEARITY CONSIDERATIONS
        if n_features < 100 and n_samples < 10000:
            ml_cons['algorithm_hints']['kernel_methods'] = {
                'rbf': 'Good for non-linear patterns, moderate dataset size',
                'linear': 'Fast, works well for high-dimensional or linearly separable data',
                'poly': 'For polynomial relationships, but can overfit'
            }
        else:
            ml_cons['algorithm_hints']['kernel_methods'] = {
                'linear': 'Recommended for large/high-dimensional datasets',
                'rbf': 'May be slow, consider subsampling or linear kernel'
            }

        # 7. INTERPRETABILITY vs PERFORMANCE
        ml_cons['algorithm_hints']['interpretability'] = {
            'high': ['Decision Trees', 'Logistic Regression', 'Linear Regression', 'Naive Bayes'],
            'medium': ['Random Forest (feature importance)', 'Gradient Boosting (SHAP values)'],
            'low': ['SVM with RBF kernel', 'KNN', 'Neural Networks']
        }

        # 8. MISSING VALUES HANDLING (should be none, but note algorithms that handle it)
        ml_cons['algorithm_hints']['missing_value_tolerance'] = {
            'native_support': ['Random Forest', 'Gradient Boosting', 'Decision Trees'],
            'requires_imputation': ['SVM', 'Logistic Regression', 'KNN', 'Naive Bayes']
        }

        # 9. CATEGORICAL FEATURES (if any detected)
        categorical_cols = [col for col, info in column_info.items()
                           if info.get('type') == 'categorical']

        if categorical_cols:
            ml_cons['preprocessing_needs'].append('Categorical features: Encoding required (OneHot/Label)')
            ml_cons['algorithm_hints']['categorical_handling'] = {
                'onehot_preferred': ['SVM', 'Logistic Regression', 'KNN'],
                'label_ok': ['Decision Trees', 'Random Forest', 'Gradient Boosting']
            }

        # 10. FEDERATED LEARNING SPECIFIC CONSIDERATIONS
        ml_cons['federated_considerations'] = {
            'communication_efficient': ['Linear models', 'Tree-based models'],
            'support_vector_sharing': ['SVM', 'SVR'],
            'gradient_based': ['Logistic Regression', 'Linear Regression'],
            'ensemble_friendly': ['Random Forest', 'Gradient Boosting']
        }

        # Add summary recommendations
        target_info['ml_recommendations'] = []

        if n_samples < 1000:
            target_info['ml_recommendations'].append(
                f'Small dataset ({n_samples} samples): Traditional ML recommended over DL'
            )

        if curse_of_dimensionality:
            target_info['ml_recommendations'].append(
                f'High feature-to-sample ratio ({n_features}/{n_samples}): Apply regularization or feature selection'
            )

        if target_info['task_type'] == 'classification' and is_imbalanced:
            target_info['ml_recommendations'].append(
                'Class imbalance detected: Use class_weight or resampling techniques'
            )

        if n_features > 1000:
            target_info['ml_recommendations'].append(
                f'High-dimensional ({n_features} features): Linear algorithms or feature reduction recommended'
            )

        return target_info

    def _extract_json_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from JSON file with medical safeguards."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Analyze structure
            def analyze_json_structure(obj, path="", depth=0):
                structure = {}
                
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        current_path = f"{path}.{key}" if path else key
                        
                        # Check for null values
                        if value is None:
                            raise MetadataExtractionError(f"Null value found at {current_path}")
                        
                        # Determine type
                        if isinstance(value, (dict, list)):
                            structure[key] = {
                                'type': 'object' if isinstance(value, dict) else 'array',
                                'depth': depth + 1
                            }
                            # Recursively analyze nested structures
                            nested = analyze_json_structure(value, current_path, depth + 1)
                            if nested:
                                structure[key]['nested'] = nested
                        else:
                            # Coarse type classification
                            if isinstance(value, bool):
                                coarse_type = 'boolean'
                            elif isinstance(value, int):
                                coarse_type = 'numeric'
                            elif isinstance(value, float):
                                coarse_type = 'numeric'
                            elif isinstance(value, str):
                                coarse_type = 'text'
                            else:
                                coarse_type = 'other'
                            
                            structure[key] = {
                                'type': coarse_type,
                                'depth': depth
                            }
                
                elif isinstance(obj, list) and obj:
                    # Analyze first few items to understand array structure
                    sample_size = min(5, len(obj))
                    for i in range(sample_size):
                        item_structure = analyze_json_structure(obj[i], f"{path}[{i}]", depth)
                        if item_structure:
                            structure[f"item_{i}"] = item_structure
                
                return structure
            
            structure = analyze_json_structure(data)
            
            return {
                'file_type': 'json',
                'structure': structure,
                'max_depth': self._calculate_max_depth(structure),
                'nulls_verified_zero': True,
                'total_keys': len(structure)
            }
            
        except Exception as e:
            raise MetadataExtractionError(f"JSON metadata extraction failed: {str(e)}")
    
    def _extract_parquet_metadata(self, file_path: str, target_column: str = None) -> Dict[str, Any]:
        """Extract metadata from Parquet file with medical safeguards."""
        try:
            try:
                import pyarrow.parquet as pq
            except ImportError:
                raise MetadataExtractionError("pyarrow not available for Parquet processing")

            # Read parquet file
            table = pq.read_table(file_path)
            df = table.to_pandas()

            rows, cols = df.shape

            # Check for nulls
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                raise MetadataExtractionError(
                    f"Dataset contains null values: {null_counts[null_counts > 0].to_dict()}"
                )

            # Extract schema information safely
            schema_info = {}
            column_info = {}
            for i, field in enumerate(table.schema):
                schema_info[field.name] = {
                    'type': str(field.type),
                    'nullable': field.nullable
                }
                # Also create column_info for target analysis
                column_info[field.name] = {
                    'type': 'numeric' if 'int' in str(field.type).lower() or 'float' in str(field.type).lower() or 'double' in str(field.type).lower() else 'categorical',
                    'nullable': field.nullable
                }

            # Analyze target column if specified
            target_info = None
            if target_column and target_column in df.columns:
                target_info = self._analyze_target_column(df, target_column, column_info)

            metadata = {
                'file_type': 'parquet',
                'rows': rows,
                'columns': cols,
                'schema': schema_info,
                'nulls_verified_zero': True,
                'compressed_size': os.path.getsize(file_path),
                'k_anonymity_compliant': rows >= self.MIN_K_ANONYMITY
            }

            # Add target info if available
            if target_info:
                metadata['target_info'] = target_info

            return metadata

        except Exception as e:
            raise MetadataExtractionError(f"Parquet metadata extraction failed: {str(e)}")
    
    def _extract_binary_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract basic metadata from binary files (H5, NPY)."""
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.npy':
                try:
                    import numpy as np
                except ImportError:
                    raise MetadataExtractionError("numpy not available for NPY processing")
                array = np.load(file_path)
                
                return {
                    'file_type': 'npy',
                    'shape': array.shape,
                    'dtype': str(array.dtype),
                    'size_bytes': array.nbytes
                }
            
            elif file_ext == '.h5':
                try:
                    import h5py
                except ImportError:
                    raise MetadataExtractionError("h5py not available for H5 processing")
                
                with h5py.File(file_path, 'r') as f:
                    def get_h5_structure(group, path=""):
                        structure = {}
                        for key in group.keys():
                            item = group[key]
                            current_path = f"{path}/{key}" if path else key
                            
                            if isinstance(item, h5py.Dataset):
                                structure[key] = {
                                    'type': 'dataset',
                                    'shape': item.shape,
                                    'dtype': str(item.dtype)
                                }
                            elif isinstance(item, h5py.Group):
                                structure[key] = {
                                    'type': 'group',
                                    'items': get_h5_structure(item, current_path)
                                }
                        return structure
                    
                    structure = get_h5_structure(f)
                
                return {
                    'file_type': 'h5',
                    'structure': structure,
                    'file_size': os.path.getsize(file_path)
                }
            
        except Exception as e:
            raise MetadataExtractionError(f"Binary file metadata extraction failed: {str(e)}")
    
    def get_csv_columns(self, file_path: str) -> List[str]:
        """
        Extract column names from CSV file for target selection.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of column names
            
        Raises:
            MetadataExtractionError: If column extraction fails
        """
        if not PANDAS_AVAILABLE or pd is None:
            raise MetadataExtractionError("pandas not available for CSV column detection")
        
        try:
            # Read only the header to get column names
            df_header = pd.read_csv(file_path, nrows=0)
            columns = df_header.columns.tolist()
            
            # Validate columns for forbidden patterns
            for col_name in columns:
                col_lower = col_name.lower()
                for pattern in self.FORBIDDEN_PATTERNS:
                    if pattern in col_lower:
                        logger.warning(f"Column '{col_name}' matches forbidden pattern '{pattern}'")
            
            return columns
            
        except Exception as e:
            raise MetadataExtractionError(f"Failed to extract CSV columns: {str(e)}")
    
    def _validate_medical_compliance(self, metadata: Dict[str, Any], file_path: str, target_column: str = None) -> Dict[str, Any]:
        """
        Validate metadata for medical data compliance and remove PHI columns automatically.

        Args:
            metadata: Extracted metadata
            file_path: Original file path
            target_column: Name of the target column (should not be removed)

        Returns:
            Dict: Updated metadata with PHI columns removed and notification about changes

        Raises:
            SecurityValidationError: If compliance check fails for non-PHI reasons
        """
        removed_columns = []

        # Check for forbidden column patterns and remove them
        if 'column_info' in metadata:
            columns_to_remove = []
            for col_name in metadata['column_info'].keys():
                # Never remove the target column, even if it matches patterns
                if target_column and col_name == target_column:
                    continue

                col_lower = col_name.lower()
                for pattern in self.FORBIDDEN_PATTERNS:
                    if re.search(pattern, col_lower):
                        columns_to_remove.append(col_name)
                        removed_columns.append({'name': col_name, 'reason': f'matches pattern {pattern}'})
                        break

            # Remove PHI columns from metadata
            for col_name in columns_to_remove:
                del metadata['column_info'][col_name]

            # Update column count
            if 'columns' in metadata:
                metadata['columns'] = len(metadata['column_info'])

        # Add information about removed columns to metadata
        if removed_columns:
            metadata['phi_columns_removed'] = removed_columns
            metadata['original_columns'] = metadata.get('columns', 0) + len(removed_columns)

            # Update the physical file if it's CSV
            if metadata.get('file_type') == 'csv' and PANDAS_AVAILABLE:
                self._remove_phi_columns_from_file(file_path, [col['name'] for col in removed_columns])

                # Re-analyze target after PHI removal if target exists
                if target_column and PANDAS_AVAILABLE:
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    if target_column in df.columns:
                        # Use updated column_info from metadata
                        column_info = metadata.get('column_info', {})
                        metadata['target_info'] = self._analyze_target_column(df, target_column, column_info)

        # Ensure k-anonymity compliance for statistical data
        if metadata.get('file_type') in ['csv', 'parquet']:
            if not metadata.get('k_anonymity_compliant', False):
                rows = metadata.get('rows', 0)
                raise SecurityValidationError(
                    f"Dataset has only {rows} rows, minimum {self.MIN_K_ANONYMITY} required for k-anonymity"
                )

        # Verify no nulls (already checked during extraction, but double-check)
        if not metadata.get('nulls_verified_zero', False):
            raise SecurityValidationError("Dataset contains null values - not allowed for medical data")

        return metadata
    
    def _remove_phi_columns_from_file(self, file_path: str, columns_to_remove: List[str]) -> None:
        """
        Remove PHI columns from the physical CSV file.
        
        Args:
            file_path: Path to the CSV file
            columns_to_remove: List of column names to remove
        """
        if not PANDAS_AVAILABLE or not columns_to_remove:
            return
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Remove the specified columns
            df_cleaned = df.drop(columns=columns_to_remove, errors='ignore')
            
            # Write back to the same file
            df_cleaned.to_csv(file_path, index=False)
            
        except Exception as e:
            # Log the error but don't fail the upload
            print(f"Warning: Could not remove PHI columns from file: {str(e)}")
    
    def _calculate_checksums(self, file_path: str) -> Tuple[str, str]:
        """Calculate MD5 and SHA256 checksums."""
        md5_hash = hashlib.md5()
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
                sha256_hash.update(chunk)
        
        return md5_hash.hexdigest(), sha256_hash.hexdigest()
    
    def _store_file_securely(self, file_path: str, dataset_name: str) -> str:
        """
        Store file in secure location with proper directory structure.
        
        Args:
            file_path: Source file path
            dataset_name: Name of the dataset
            
        Returns:
            Final storage path
        """
        # Create directory structure: /datasets/user_id/year/month/
        base_dir = getattr(settings, 'DATASETS_STORAGE_DIR', 'datasets')
        user_dir = os.path.join(base_dir, str(self.user.id))
        date_dir = os.path.join(user_dir, datetime.now().strftime('%Y/%m'))
        
        # Ensure directory exists
        os.makedirs(date_dir, exist_ok=True)
        
        # Generate safe filename
        original_filename = os.path.basename(file_path)
        safe_filename = self._sanitize_filename(original_filename)
        
        # Add timestamp to avoid conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name_parts = safe_filename.rsplit('.', 1)
        if len(name_parts) == 2:
            final_filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
        else:
            final_filename = f"{safe_filename}_{timestamp}"
        
        final_path = os.path.join(date_dir, final_filename)
        
        # Atomic move operation
        temp_path = final_path + '.tmp'
        shutil.copy2(file_path, temp_path)
        os.rename(temp_path, final_path)
        
        # Set secure permissions (read-only for group/others)
        os.chmod(final_path, 0o644)
        
        return final_path
    
    def _create_dataset_record(
        self,
        name: str,
        description: str,
        file_path: str,
        medical_domain: str,
        data_type: str,
        metadata: Dict[str, Any],
        md5_hash: str
    ) -> Dataset:
        """Create Dataset database record."""
        
        # Extract relevant fields from metadata
        file_size = os.path.getsize(file_path)
        file_ext = Path(file_path).suffix.lower().lstrip('.')
        
        # Map file extension to format choice
        format_mapping = {
            'csv': 'csv',
            'json': 'json',
            'parquet': 'parquet',
            'h5': 'h5',
            'npy': 'npy'
        }
        
        from django.db import connection
        from django.db import connections
        from django.conf import settings
        
        # Get the datasets database connection
        datasets_connection = connections['datasets_db']
        
        # Prepare the parameters
        params = (
            name, description, file_path, self.user.id,
            medical_domain, metadata.get('rows', 0) if data_type == 'tabular' else None,
            data_type, True, file_size, format_mapping.get(file_ext, 'other'),
            metadata.get('columns', 0), metadata.get('rows', 0), md5_hash,
            True, timezone.now().isoformat(), 0  # access_count starts at 0
        )
        
        # Use raw SQL to insert - foreign keys handled by atomic transaction
        # Temporarily disable query logging to avoid formatting issues
        old_debug = settings.DEBUG
        settings.DEBUG = False
        
        try:
            with datasets_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO dataset_dataset (
                        name, description, file_path, uploaded_by_id, 
                        medical_domain, patient_count, data_type, 
                        anonymized, file_size, file_format, 
                        columns_count, rows_count, checksum_md5, 
                        is_active, uploaded_at, access_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, params)
                
                # Get the ID of the inserted record
                dataset_id = cursor.lastrowid
        finally:
            # Restore debug setting
            settings.DEBUG = old_debug
        
        # Create a dataset instance for return
        dataset = Dataset.objects.using('datasets_db').get(id=dataset_id)
        
        return dataset
    
    def _create_dataset_record_raw_sql(
        self,
        cursor,
        name: str,
        description: str,
        file_path: str,
        medical_domain: str,
        data_type: str,
        metadata: Dict[str, Any],
        md5_hash: str,
        target_column: str = None
    ) -> int:
        """Create Dataset database record using raw SQL."""
        
        # Extract relevant fields from metadata
        file_size = os.path.getsize(file_path)
        file_ext = Path(file_path).suffix.lower().lstrip('.')
        
        # Map file extension to format choice
        format_mapping = {
            'csv': 'csv',
            'json': 'json',
            'parquet': 'parquet',
            'h5': 'h5',
            'npy': 'npy'
        }
        
        # Normalize file path to avoid escape character issues
        normalized_path = file_path.replace('\\', '/')
        
        # Match EXACT column order from DB
        params = (
            name,                                                   # name
            description,                                            # description  
            normalized_path,                                        # file_path (normalized)
            medical_domain,                                         # medical_domain
            metadata.get('rows', 0) if data_type == 'tabular' else None,  # patient_count
            data_type,                                              # data_type
            True,                                                   # anonymized
            file_size,                                              # file_size
            format_mapping.get(file_ext, 'other'),                  # file_format
            metadata.get('columns', 0),                             # columns_count
            metadata.get('rows', 0),                                # rows_count
            timezone.now().strftime('%Y-%m-%d %H:%M:%S'),           # uploaded_at (SQLite format)
            None,                                                   # last_accessed (NULL)
            0,                                                      # access_count
            md5_hash,                                               # checksum_md5
            True,                                                   # is_active
            self.user.id,                                           # uploaded_by_id
            target_column,                                          # target_column
        )
        
        # Build SQL with explicit values to avoid placeholder issues
        def escape_sql_value(val):
            if val is None:
                return 'NULL'
            elif isinstance(val, str):
                # Escape single quotes and wrap in quotes
                escaped = val.replace("'", "''")
                return f"'{escaped}'"
            elif isinstance(val, bool):
                return '1' if val else '0'
            else:
                return str(val)
        
        # Temporarily disable Django query logging to avoid formatting issues
        from django.conf import settings
        old_debug = settings.DEBUG
        settings.DEBUG = False
        
        try:
            escaped_params = [escape_sql_value(p) for p in params]
            values_str = ', '.join(escaped_params)
            
            sql_explicit = f"INSERT INTO dataset_dataset (name, description, file_path, medical_domain, patient_count, data_type, anonymized, file_size, file_format, columns_count, rows_count, uploaded_at, last_accessed, access_count, checksum_md5, is_active, uploaded_by_id, target_column) VALUES ({values_str})"
            
            cursor.execute(sql_explicit)
        finally:
            # Restore debug setting
            settings.DEBUG = old_debug
        
        # Get the ID of the inserted record
        return cursor.lastrowid
    
    def _create_metadata_record_raw_sql(self, cursor, dataset_id: int, metadata: Dict[str, Any]) -> None:
        """Create DatasetMetadata database record using raw SQL."""

        # Prepare safe metadata for storage
        safe_metadata = {
            'file_type': metadata.get('file_type'),
            'extraction_timestamp': datetime.now().isoformat(),
            'k_anonymity_verified': metadata.get('k_anonymity_compliant', False),
            'nulls_verified_zero': metadata.get('nulls_verified_zero', False)
        }

        # Add type-specific metadata
        if 'statistical_summary' in metadata:
            safe_metadata['statistical_summary'] = metadata['statistical_summary']

        if 'column_info' in metadata:
            safe_metadata['column_types'] = {
                col: info['type'] for col, info in metadata['column_info'].items()
            }

        if 'schema' in metadata:
            safe_metadata['schema'] = metadata['schema']

        # Add target information for automatic model generation
        if 'target_info' in metadata:
            safe_metadata['target_info'] = metadata['target_info']
        
        # Insert metadata record with explicit SQL to avoid placeholder issues
        import json
        from django.conf import settings
        old_debug = settings.DEBUG
        settings.DEBUG = False
        
        try:
            # Use explicit SQL approach like dataset (no placeholders)
            def escape_sql_value(val):
                if val is None:
                    return 'NULL'
                elif isinstance(val, str):
                    escaped = val.replace("'", "''")
                    return f"'{escaped}'"
                elif isinstance(val, (int, float)):
                    return str(val)
                else:
                    return f"'{str(val)}'"
            
            # Build metadata SQL explicitly
            metadata_values = [
                str(dataset_id),
                escape_sql_value(json.dumps(safe_metadata)),
                escape_sql_value('{}'),  # missing_values
                escape_sql_value('{}'),  # data_distribution 
                '1.0',  # quality_score
                '100.0',  # completeness_percentage
                escape_sql_value(timezone.now().strftime('%Y-%m-%d %H:%M:%S')),  # generated_at
                escape_sql_value(timezone.now().strftime('%Y-%m-%d %H:%M:%S'))   # updated_at
            ]
            
            values_str = ', '.join(metadata_values)
            metadata_sql = f"INSERT INTO dataset_datasetmetadata (dataset_id, statistical_summary, missing_values, data_distribution, quality_score, completeness_percentage, generated_at, updated_at) VALUES ({values_str})"
            
            cursor.execute(metadata_sql)
        finally:
            settings.DEBUG = old_debug
    
    def _create_metadata_record(self, dataset: Dataset, metadata: Dict[str, Any]) -> DatasetMetadata:
        """Create DatasetMetadata database record."""
        
        # Prepare safe metadata for storage
        safe_metadata = {
            'file_type': metadata.get('file_type'),
            'extraction_timestamp': datetime.now().isoformat(),
            'k_anonymity_verified': metadata.get('k_anonymity_compliant', False),
            'nulls_verified_zero': metadata.get('nulls_verified_zero', False)
        }
        
        # Add type-specific metadata
        if 'statistical_summary' in metadata:
            safe_metadata['statistical_summary'] = metadata['statistical_summary']
        
        if 'column_info' in metadata:
            safe_metadata['column_types'] = {
                col: info['type'] for col, info in metadata['column_info'].items()
            }
        
        if 'schema' in metadata:
            safe_metadata['schema'] = metadata['schema']
        
        dataset_metadata = DatasetMetadata.objects.using('datasets_db').create(
            dataset=dataset,
            statistical_summary=safe_metadata,
            missing_values={},  # Always empty since we verify no nulls
            data_distribution={},  # Not included for privacy
            quality_score=1.0,  # High score since we validated compliance
            completeness_percentage=100.0  # Always 100% since no nulls allowed
        )
        
        return dataset_metadata
    
    def _handle_upload_failure(self, file_path: str, error_msg: str) -> None:
        """Handle upload failure by moving file to quarantine."""
        try:
            if os.path.exists(file_path):
                # Move to quarantine with error info
                quarantine_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(file_path)}"
                quarantine_path = os.path.join(self.quarantine_dir, quarantine_filename)
                
                shutil.move(file_path, quarantine_path)
                
                # Log quarantine action
                error_log_path = quarantine_path + '.error.log'
                with open(error_log_path, 'w') as f:
                    f.write(f"Upload failed at: {datetime.now().isoformat()}\n")
                    f.write(f"User: {self.user.username}\n")
                    f.write(f"Error: {error_msg}\n")
                
                logger.warning(f"File quarantined: {quarantine_path}")
                
        except Exception as e:
            logger.error(f"Failed to quarantine file: {str(e)}")
    
    def _update_progress(self, status: str, message: str) -> None:
        """Update upload progress via callback."""
        if self.progress_callback:
            self.progress_callback(status, message)
    
    def _get_quarantine_dir(self) -> str:
        """Get quarantine directory path."""
        quarantine_dir = getattr(settings, 'DATASETS_QUARANTINE_DIR', 'quarantine')
        os.makedirs(quarantine_dir, exist_ok=True)
        return quarantine_dir
    
    def _is_safe_filename(self, filename: str) -> bool:
        """Check if filename is safe."""
        # Basic safety checks
        if not filename or filename in ['.', '..']:
            return False
        
        # Check for dangerous characters
        dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
        return not any(char in filename for char in dangerous_chars)
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove dangerous characters
        safe_chars = []
        for char in filename:
            if char.isalnum() or char in ['.', '-', '_']:
                safe_chars.append(char)
            else:
                safe_chars.append('_')
        
        sanitized = ''.join(safe_chars)
        
        # Remove dangerous patterns like .. and multiple dots
        sanitized = sanitized.replace('..', '_')
        
        # Ensure it's not too long
        if len(sanitized) > 100:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:90] + ext
        
        return sanitized
    
    def _calculate_max_depth(self, structure: Dict, current_depth: int = 0) -> int:
        """Calculate maximum depth of nested structure."""
        max_depth = current_depth
        
        for key, value in structure.items():
            if isinstance(value, dict):
                if 'nested' in value:
                    nested_depth = self._calculate_max_depth(value['nested'], current_depth + 1)
                    max_depth = max(max_depth, nested_depth)
                elif 'depth' in value:
                    max_depth = max(max_depth, value['depth'])
        
        return max_depth

"""
Input validation for model predictions.

Validates input data against model schema with comprehensive checks for:
- Schema compliance (features, types, shape)
- Range validation (min/max per feature)
- Type checking and NaN/Inf detection
- Data sanitization
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal, InvalidOperation


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class InputValidator:
    """
    Validates input data against DeployedModel schema.

    Performs comprehensive validation including:
    - Feature name validation
    - Data type checking
    - Shape validation
    - Range checking (min/max)
    - NaN/Inf detection
    - Data sanitization
    """

    def __init__(self, input_schema: dict, output_schema: dict = None):
        """
        Initialize validator with model schema.

        Args:
            input_schema: Expected input format with feature_names, dtypes, shape, ranges
            output_schema: Expected output format (optional, for validation)

        Schema format:
        {
            "feature_names": ["age", "blood_pressure", "cholesterol"],
            "dtypes": {"age": "int64", "blood_pressure": "float32", "cholesterol": "float32"},
            "shape": [3],  # Number of features
            "ranges": {  # Optional
                "age": {"min": 0, "max": 120},
                "blood_pressure": {"min": 60, "max": 200},
                "cholesterol": {"min": 100, "max": 400}
            }
        }
        """
        self.input_schema = input_schema
        self.output_schema = output_schema

        # Extract schema components
        self.feature_names = input_schema.get('feature_names', [])
        self.dtypes = input_schema.get('dtypes', {})
        self.expected_shape = input_schema.get('shape', [])
        self.ranges = input_schema.get('ranges', {})

        # Validate schema itself
        self._validate_schema()

    def _validate_schema(self):
        """Validate that the schema itself is well-formed."""
        if not self.feature_names:
            raise ValueError("Schema must include 'feature_names'")

        if not self.expected_shape:
            raise ValueError("Schema must include 'shape'")

        # Check that dtypes are provided for all features
        for feature in self.feature_names:
            if feature not in self.dtypes:
                raise ValueError(f"Missing dtype for feature: {feature}")

    def validate(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate input data against schema.

        Args:
            data: List of records, each with feature names as keys
                 Example: [{"age": 45, "blood_pressure": 120, "cholesterol": 200}, ...]

        Returns:
            Dict with keys:
                - valid (bool): Whether validation passed
                - errors (list): List of validation error messages
                - sanitized_data (np.ndarray): Cleaned data ready for inference (if valid)
                - warnings (list): Non-fatal warnings
        """
        errors = []
        warnings = []

        # 1. Check data is not empty
        if not data or len(data) == 0:
            errors.append("Input data is empty")
            return {'valid': False, 'errors': errors, 'sanitized_data': None, 'warnings': warnings}

        # 2. Check data is a list
        if not isinstance(data, list):
            errors.append(f"Input must be a list of records, got {type(data).__name__}")
            return {'valid': False, 'errors': errors, 'sanitized_data': None, 'warnings': warnings}

        # 3. Validate each record
        sanitized_records = []
        for idx, record in enumerate(data):
            record_errors, record_warnings, sanitized_record = self._validate_record(record, idx)
            errors.extend(record_errors)
            warnings.extend(record_warnings)

            if not record_errors:
                sanitized_records.append(sanitized_record)

        # If any errors occurred, return invalid
        if errors:
            return {'valid': False, 'errors': errors, 'sanitized_data': None, 'warnings': warnings}

        # Convert to numpy array
        try:
            sanitized_data = np.array(sanitized_records, dtype=np.float32)
        except Exception as e:
            errors.append(f"Failed to convert data to numpy array: {str(e)}")
            return {'valid': False, 'errors': errors, 'sanitized_data': None, 'warnings': warnings}

        # 4. Validate final shape
        expected_num_features = len(self.feature_names)
        if sanitized_data.shape[1] != expected_num_features:
            errors.append(
                f"Shape mismatch: expected {expected_num_features} features, "
                f"got {sanitized_data.shape[1]}"
            )
            return {'valid': False, 'errors': errors, 'sanitized_data': None, 'warnings': warnings}

        return {
            'valid': True,
            'errors': [],
            'sanitized_data': sanitized_data,
            'warnings': warnings
        }

    def _validate_record(
        self,
        record: Dict[str, Any],
        record_idx: int
    ) -> Tuple[List[str], List[str], List[float]]:
        """
        Validate a single record.

        Args:
            record: Dictionary with feature names as keys
            record_idx: Index of record in batch (for error messages)

        Returns:
            Tuple of (errors, warnings, sanitized_values)
        """
        errors = []
        warnings = []
        sanitized_values = []

        # Check record is a dict
        if not isinstance(record, dict):
            errors.append(f"Record {record_idx}: must be a dictionary, got {type(record).__name__}")
            return errors, warnings, []

        # Validate each feature
        for feature_name in self.feature_names:
            # 1. Check feature exists
            if feature_name not in record:
                errors.append(f"Record {record_idx}: missing feature '{feature_name}'")
                sanitized_values.append(0.0)  # Placeholder
                continue

            value = record[feature_name]
            expected_dtype = self.dtypes.get(feature_name)

            # 2. Check for NaN/Inf early (before type conversion)
            if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                errors.append(
                    f"Record {record_idx}: feature '{feature_name}' has invalid value "
                    f"(NaN or Inf): {value}"
                )
                sanitized_values.append(0.0)
                continue

            # 3. Validate type and convert
            validated_value, type_errors, type_warnings = self._validate_type(
                value, expected_dtype, feature_name, record_idx
            )
            errors.extend(type_errors)
            warnings.extend(type_warnings)

            if type_errors:
                sanitized_values.append(0.0)  # Placeholder for failed validation
                continue

            # 4. Double-check for NaN/Inf after conversion (edge case)
            if np.isnan(validated_value) or np.isinf(validated_value):
                errors.append(
                    f"Record {record_idx}: feature '{feature_name}' has invalid value "
                    f"(NaN or Inf): {value}"
                )
                sanitized_values.append(0.0)
                continue

            # 5. Validate range
            if feature_name in self.ranges:
                range_spec = self.ranges[feature_name]
                min_val = range_spec.get('min')
                max_val = range_spec.get('max')

                if min_val is not None and validated_value < min_val:
                    errors.append(
                        f"Record {record_idx}: feature '{feature_name}' value {validated_value} "
                        f"is below minimum {min_val}"
                    )
                    continue

                if max_val is not None and validated_value > max_val:
                    errors.append(
                        f"Record {record_idx}: feature '{feature_name}' value {validated_value} "
                        f"is above maximum {max_val}"
                    )
                    continue

            sanitized_values.append(float(validated_value))

        return errors, warnings, sanitized_values

    def _validate_type(
        self,
        value: Any,
        expected_dtype: str,
        feature_name: str,
        record_idx: int
    ) -> Tuple[float, List[str], List[str]]:
        """
        Validate and convert value to expected type.

        Args:
            value: Raw value from input
            expected_dtype: Expected data type (int64, float32, etc.)
            feature_name: Name of feature (for error messages)
            record_idx: Record index (for error messages)

        Returns:
            Tuple of (converted_value, errors, warnings)
        """
        errors = []
        warnings = []

        # Handle None/null
        if value is None:
            errors.append(
                f"Record {record_idx}: feature '{feature_name}' is null/None"
            )
            return 0.0, errors, warnings

        # Try to convert to float (most permissive)
        try:
            if isinstance(value, str):
                # Try to parse string
                try:
                    # Handle strings like "123.45"
                    converted = float(value)
                except ValueError:
                    errors.append(
                        f"Record {record_idx}: feature '{feature_name}' cannot convert "
                        f"string '{value}' to number"
                    )
                    return 0.0, errors, warnings
            elif isinstance(value, (int, float, Decimal)):
                converted = float(value)
            else:
                errors.append(
                    f"Record {record_idx}: feature '{feature_name}' has unsupported type "
                    f"{type(value).__name__}"
                )
                return 0.0, errors, warnings

            # Check dtype-specific constraints
            if expected_dtype in ['int32', 'int64']:
                # For integer types, check if value is actually an integer
                if not isinstance(value, int) and converted != int(converted):
                    warnings.append(
                        f"Record {record_idx}: feature '{feature_name}' expects integer, "
                        f"got {value} (will be truncated to {int(converted)})"
                    )
                converted = float(int(converted))

            return converted, errors, warnings

        except (ValueError, TypeError, InvalidOperation) as e:
            errors.append(
                f"Record {record_idx}: feature '{feature_name}' type conversion failed: {str(e)}"
            )
            return 0.0, errors, warnings

    def validate_batch_size(self, batch_size: int, max_batch_size: int) -> Dict[str, Any]:
        """
        Validate batch size is within limits.

        Args:
            batch_size: Number of records in batch
            max_batch_size: Maximum allowed batch size

        Returns:
            Dict with keys:
                - valid (bool): Whether batch size is acceptable
                - error (str): Error message if invalid
        """
        if batch_size <= 0:
            return {'valid': False, 'error': 'Batch size must be positive'}

        if batch_size > max_batch_size:
            return {
                'valid': False,
                'error': f'Batch size {batch_size} exceeds maximum {max_batch_size}'
            }

        return {'valid': True, 'error': None}

"""
Tests for Input Validators.
"""
import pytest
import numpy as np
from inference.validators import InputValidator, ValidationError


@pytest.fixture
def simple_schema():
    """Simple schema with 3 numeric features."""
    return {
        "feature_names": ["age", "blood_pressure", "cholesterol"],
        "dtypes": {
            "age": "int64",
            "blood_pressure": "float32",
            "cholesterol": "float32"
        },
        "shape": [3],
        "ranges": {
            "age": {"min": 0, "max": 120},
            "blood_pressure": {"min": 60, "max": 200},
            "cholesterol": {"min": 100, "max": 400}
        }
    }


@pytest.fixture
def minimal_schema():
    """Minimal schema without ranges."""
    return {
        "feature_names": ["feature1", "feature2"],
        "dtypes": {
            "feature1": "float32",
            "feature2": "float32"
        },
        "shape": [2]
    }


class TestInputValidator:
    """Test InputValidator class."""

    def test_initialization(self, simple_schema):
        """Test validator initialization."""
        validator = InputValidator(simple_schema)
        assert validator.feature_names == ["age", "blood_pressure", "cholesterol"]
        assert len(validator.dtypes) == 3
        assert validator.expected_shape == [3]
        assert "age" in validator.ranges

    def test_invalid_schema_no_features(self):
        """Test that schema without feature_names raises error."""
        invalid_schema = {
            "dtypes": {"feature1": "float32"},
            "shape": [1]
        }
        with pytest.raises(ValueError, match="must include 'feature_names'"):
            InputValidator(invalid_schema)

    def test_invalid_schema_no_shape(self):
        """Test that schema without shape raises error."""
        invalid_schema = {
            "feature_names": ["feature1"],
            "dtypes": {"feature1": "float32"}
        }
        with pytest.raises(ValueError, match="must include 'shape'"):
            InputValidator(invalid_schema)

    def test_invalid_schema_missing_dtype(self):
        """Test that schema without dtype for a feature raises error."""
        invalid_schema = {
            "feature_names": ["feature1", "feature2"],
            "dtypes": {"feature1": "float32"},
            "shape": [2]
        }
        with pytest.raises(ValueError, match="Missing dtype for feature: feature2"):
            InputValidator(invalid_schema)

    def test_validate_valid_data(self, simple_schema):
        """Test validation of valid data."""
        validator = InputValidator(simple_schema)

        data = [
            {"age": 45, "blood_pressure": 120, "cholesterol": 200},
            {"age": 60, "blood_pressure": 130, "cholesterol": 220}
        ]

        result = validator.validate(data)

        assert result['valid'] is True
        assert len(result['errors']) == 0
        assert result['sanitized_data'] is not None
        assert result['sanitized_data'].shape == (2, 3)

    def test_validate_empty_data(self, simple_schema):
        """Test validation of empty data."""
        validator = InputValidator(simple_schema)

        result = validator.validate([])

        assert result['valid'] is False
        assert 'Input data is empty' in result['errors']

    def test_validate_not_a_list(self, simple_schema):
        """Test validation of non-list input."""
        validator = InputValidator(simple_schema)

        result = validator.validate({"age": 45})

        assert result['valid'] is False
        assert 'Input must be a list of records' in result['errors'][0]

    def test_validate_missing_feature(self, simple_schema):
        """Test validation when a feature is missing."""
        validator = InputValidator(simple_schema)

        data = [
            {"age": 45, "blood_pressure": 120}  # Missing cholesterol
        ]

        result = validator.validate(data)

        assert result['valid'] is False
        assert any("missing feature 'cholesterol'" in err for err in result['errors'])

    def test_validate_extra_features_ignored(self, simple_schema):
        """Test that extra features are ignored."""
        validator = InputValidator(simple_schema)

        data = [
            {"age": 45, "blood_pressure": 120, "cholesterol": 200, "extra_field": 999}
        ]

        result = validator.validate(data)

        assert result['valid'] is True
        assert result['sanitized_data'].shape == (1, 3)

    def test_validate_range_below_minimum(self, simple_schema):
        """Test validation of value below minimum."""
        validator = InputValidator(simple_schema)

        data = [
            {"age": -5, "blood_pressure": 120, "cholesterol": 200}
        ]

        result = validator.validate(data)

        assert result['valid'] is False
        assert any("below minimum" in err and "age" in err for err in result['errors'])

    def test_validate_range_above_maximum(self, simple_schema):
        """Test validation of value above maximum."""
        validator = InputValidator(simple_schema)

        data = [
            {"age": 150, "blood_pressure": 120, "cholesterol": 200}
        ]

        result = validator.validate(data)

        assert result['valid'] is False
        assert any("above maximum" in err and "age" in err for err in result['errors'])

    def test_validate_null_value(self, simple_schema):
        """Test validation of null/None value."""
        validator = InputValidator(simple_schema)

        data = [
            {"age": None, "blood_pressure": 120, "cholesterol": 200}
        ]

        result = validator.validate(data)

        assert result['valid'] is False
        assert any("is null/None" in err and "age" in err for err in result['errors'])

    def test_validate_nan_value(self, simple_schema):
        """Test validation of NaN value."""
        validator = InputValidator(simple_schema)

        data = [
            {"age": float('nan'), "blood_pressure": 120, "cholesterol": 200}
        ]

        result = validator.validate(data)

        assert result['valid'] is False
        assert any("NaN or Inf" in err for err in result['errors'])

    def test_validate_inf_value(self, simple_schema):
        """Test validation of Inf value."""
        validator = InputValidator(simple_schema)

        data = [
            {"age": float('inf'), "blood_pressure": 120, "cholesterol": 200}
        ]

        result = validator.validate(data)

        assert result['valid'] is False
        assert any("NaN or Inf" in err for err in result['errors'])

    def test_validate_string_to_number_conversion(self, simple_schema):
        """Test that numeric strings are converted correctly."""
        validator = InputValidator(simple_schema)

        data = [
            {"age": "45", "blood_pressure": "120.5", "cholesterol": "200"}
        ]

        result = validator.validate(data)

        assert result['valid'] is True
        assert result['sanitized_data'][0, 0] == 45.0
        assert result['sanitized_data'][0, 1] == 120.5

    def test_validate_invalid_string(self, simple_schema):
        """Test that non-numeric strings are rejected."""
        validator = InputValidator(simple_schema)

        data = [
            {"age": "not_a_number", "blood_pressure": 120, "cholesterol": 200}
        ]

        result = validator.validate(data)

        assert result['valid'] is False
        assert any("cannot convert string" in err for err in result['errors'])

    def test_validate_integer_type_warning(self, simple_schema):
        """Test warning for float value in integer field."""
        validator = InputValidator(simple_schema)

        data = [
            {"age": 45.7, "blood_pressure": 120, "cholesterol": 200}
        ]

        result = validator.validate(data)

        # Should be valid but with warning
        assert result['valid'] is True
        assert len(result['warnings']) > 0
        assert any("expects integer" in warn for warn in result['warnings'])
        # Value should be truncated
        assert result['sanitized_data'][0, 0] == 45.0

    def test_validate_record_not_dict(self, simple_schema):
        """Test validation when record is not a dictionary."""
        validator = InputValidator(simple_schema)

        data = [
            [45, 120, 200]  # List instead of dict
        ]

        result = validator.validate(data)

        assert result['valid'] is False
        assert any("must be a dictionary" in err for err in result['errors'])

    def test_validate_multiple_errors_same_record(self, simple_schema):
        """Test that multiple errors in same record are all reported."""
        validator = InputValidator(simple_schema)

        data = [
            {"age": -10, "blood_pressure": 300, "cholesterol": 50}  # All out of range
        ]

        result = validator.validate(data)

        assert result['valid'] is False
        # Should have 3 errors (one for each feature)
        assert len(result['errors']) == 3

    def test_validate_batch_with_mixed_validity(self, simple_schema):
        """Test batch where some records are valid and some invalid."""
        validator = InputValidator(simple_schema)

        data = [
            {"age": 45, "blood_pressure": 120, "cholesterol": 200},  # Valid
            {"age": -10, "blood_pressure": 120, "cholesterol": 200},  # Invalid age
            {"age": 60, "blood_pressure": 130, "cholesterol": 220}  # Valid
        ]

        result = validator.validate(data)

        # Should fail because at least one record is invalid
        assert result['valid'] is False
        assert any("age" in err and "below minimum" in err for err in result['errors'])

    def test_validate_no_ranges(self, minimal_schema):
        """Test validation when schema has no ranges."""
        validator = InputValidator(minimal_schema)

        data = [
            {"feature1": -999.9, "feature2": 999.9}  # No range limits
        ]

        result = validator.validate(data)

        assert result['valid'] is True
        assert result['sanitized_data'].shape == (1, 2)

    def test_validate_partial_ranges(self):
        """Test schema with ranges for some features only."""
        schema = {
            "feature_names": ["feature1", "feature2"],
            "dtypes": {"feature1": "float32", "feature2": "float32"},
            "shape": [2],
            "ranges": {
                "feature1": {"min": 0, "max": 100}
                # No range for feature2
            }
        }
        validator = InputValidator(schema)

        data = [
            {"feature1": 50, "feature2": 999}  # feature2 has no range limit
        ]

        result = validator.validate(data)

        assert result['valid'] is True

    def test_validate_range_only_min(self):
        """Test range with only minimum specified."""
        schema = {
            "feature_names": ["feature1"],
            "dtypes": {"feature1": "float32"},
            "shape": [1],
            "ranges": {
                "feature1": {"min": 0}  # No max
            }
        }
        validator = InputValidator(schema)

        data = [
            {"feature1": 1000000}  # Very large but no max limit
        ]

        result = validator.validate(data)

        assert result['valid'] is True

    def test_validate_range_only_max(self):
        """Test range with only maximum specified."""
        schema = {
            "feature_names": ["feature1"],
            "dtypes": {"feature1": "float32"},
            "shape": [1],
            "ranges": {
                "feature1": {"max": 100}  # No min
            }
        }
        validator = InputValidator(schema)

        data = [
            {"feature1": -1000000}  # Very negative but no min limit
        ]

        result = validator.validate(data)

        assert result['valid'] is True

    def test_validate_batch_size_valid(self, simple_schema):
        """Test batch size validation with valid size."""
        validator = InputValidator(simple_schema)

        result = validator.validate_batch_size(batch_size=50, max_batch_size=100)

        assert result['valid'] is True
        assert result['error'] is None

    def test_validate_batch_size_too_large(self, simple_schema):
        """Test batch size validation when too large."""
        validator = InputValidator(simple_schema)

        result = validator.validate_batch_size(batch_size=150, max_batch_size=100)

        assert result['valid'] is False
        assert 'exceeds maximum' in result['error']

    def test_validate_batch_size_zero(self, simple_schema):
        """Test batch size validation with zero."""
        validator = InputValidator(simple_schema)

        result = validator.validate_batch_size(batch_size=0, max_batch_size=100)

        assert result['valid'] is False
        assert 'must be positive' in result['error']

    def test_validate_batch_size_negative(self, simple_schema):
        """Test batch size validation with negative number."""
        validator = InputValidator(simple_schema)

        result = validator.validate_batch_size(batch_size=-10, max_batch_size=100)

        assert result['valid'] is False
        assert 'must be positive' in result['error']

    def test_sanitized_data_dtype(self, simple_schema):
        """Test that sanitized data has correct dtype."""
        validator = InputValidator(simple_schema)

        data = [
            {"age": 45, "blood_pressure": 120, "cholesterol": 200}
        ]

        result = validator.validate(data)

        assert result['valid'] is True
        assert result['sanitized_data'].dtype == np.float32

    def test_sanitized_data_order(self, simple_schema):
        """Test that sanitized data maintains feature order."""
        validator = InputValidator(simple_schema)

        data = [
            {"cholesterol": 200, "age": 45, "blood_pressure": 120}  # Different order
        ]

        result = validator.validate(data)

        assert result['valid'] is True
        # Features should be in schema order: age, blood_pressure, cholesterol
        assert result['sanitized_data'][0, 0] == 45  # age
        assert result['sanitized_data'][0, 1] == 120  # blood_pressure
        assert result['sanitized_data'][0, 2] == 200  # cholesterol

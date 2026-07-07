"""
Tests for model upload form (Task 4.3).
"""
import json
import io
from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from inference.forms import ModelUploadForm
from inference.models import DeployedModel


class ModelUploadFormTest(TestCase):
    """Test ModelUploadForm validation and processing."""

    def setUp(self):
        """Set up test data."""
        self.valid_input_schema = {
            "features": [
                {
                    "name": "age",
                    "type": "integer",
                    "min": 0,
                    "max": 120,
                    "required": True,
                    "description": "Patient age in years"
                },
                {
                    "name": "blood_pressure",
                    "type": "float",
                    "min": 60,
                    "max": 200,
                    "required": True,
                    "description": "Systolic blood pressure"
                }
            ]
        }

        self.valid_output_schema = {
            "type": "classification",
            "classes": ["no_risk", "at_risk", "high_risk"],
            "confidence_threshold": 0.75
        }

        # Create a minimal valid ONNX file (mock)
        self.onnx_content = b'ONNX_MODEL_CONTENT_HERE'
        self.onnx_file = SimpleUploadedFile(
            "test_model.onnx",
            self.onnx_content,
            content_type="application/octet-stream"
        )

    def test_form_fields_present(self):
        """Test that all required fields are present in the form."""
        form = ModelUploadForm()

        self.assertIn('name', form.fields)
        self.assertIn('version', form.fields)
        self.assertIn('domain', form.fields)
        self.assertIn('description', form.fields)
        self.assertIn('model_file', form.fields)
        self.assertIn('input_schema', form.fields)
        self.assertIn('output_schema', form.fields)
        self.assertIn('accuracy_percent', form.fields)
        self.assertIn('is_public', form.fields)

    def test_accuracy_field_range(self):
        """Test accuracy field accepts 0-100 range."""
        form_data = {
            'name': 'Test Model',
            'version': '1.0.0',
            'domain': 'cardiology',
            'description': 'Test description',
            'input_schema': json.dumps(self.valid_input_schema),
            'output_schema': json.dumps(self.valid_output_schema),
            'accuracy_percent': 94.5,
            'is_public': False
        }

        form = ModelUploadForm(data=form_data, files={'model_file': self.onnx_file})

        # Note: ONNX validation will fail with mock file, but accuracy field should be valid
        self.assertNotIn('accuracy_percent', form.errors)

    def test_accuracy_field_out_of_range(self):
        """Test accuracy field rejects values > 100."""
        form_data = {
            'name': 'Test Model',
            'version': '1.0.0',
            'domain': 'cardiology',
            'description': 'Test description',
            'input_schema': json.dumps(self.valid_input_schema),
            'output_schema': json.dumps(self.valid_output_schema),
            'accuracy_percent': 150.0,  # Invalid
            'is_public': False
        }

        form = ModelUploadForm(data=form_data, files={'model_file': self.onnx_file})

        self.assertIn('accuracy_percent', form.errors)

    def test_input_schema_validation_missing_features(self):
        """Test input schema validation rejects schema without 'features' key."""
        invalid_schema = {"invalid_key": []}

        form_data = {
            'name': 'Test Model',
            'version': '1.0.0',
            'domain': 'cardiology',
            'description': 'Test description',
            'input_schema': json.dumps(invalid_schema),
            'output_schema': json.dumps(self.valid_output_schema),
            'is_public': False
        }

        form = ModelUploadForm(data=form_data, files={'model_file': self.onnx_file})
        form.is_valid()

        self.assertIn('input_schema', form.errors)
        self.assertIn('features', str(form.errors['input_schema']))

    def test_input_schema_validation_missing_required_fields(self):
        """Test input schema validation requires at minimum the 'name' field."""
        # Schema with feature missing 'name' field should fail
        invalid_schema = {
            "features": [
                {
                    "type": "float",
                    "required": True
                    # Missing 'name' - this is required
                }
            ]
        }

        form_data = {
            'name': 'Test Model',
            'version': '1.0.0',
            'domain': 'cardiology',
            'description': 'Test description',
            'input_schema': json.dumps(invalid_schema),
            'output_schema': json.dumps(self.valid_output_schema),
            'is_public': False
        }

        form = ModelUploadForm(data=form_data, files={'model_file': self.onnx_file})
        form.is_valid()

        self.assertIn('input_schema', form.errors)
        self.assertIn('name', str(form.errors['input_schema']))

    def test_input_schema_validation_defaults_type_and_required(self):
        """Test that 'type' and 'required' fields have sensible defaults."""
        # Schema with only 'name' should be valid (type defaults to float, required to True)
        minimal_schema = {
            "features": [
                {
                    "name": "age"
                    # 'type' and 'required' should default
                }
            ]
        }

        form_data = {
            'name': 'Test Model',
            'version': '1.0.0',
            'domain': 'cardiology',
            'description': 'Test description',
            'input_schema': json.dumps(minimal_schema),
            'output_schema': json.dumps(self.valid_output_schema),
            'is_public': False
        }

        form = ModelUploadForm(data=form_data, files={'model_file': self.onnx_file})
        form.is_valid()

        # input_schema should be valid (no errors for it)
        self.assertNotIn('input_schema', form.errors)

    def test_input_schema_validation_invalid_type(self):
        """Test input schema validation rejects invalid feature types."""
        invalid_schema = {
            "features": [
                {
                    "name": "age",
                    "type": "invalid_type",  # Not in ['integer', 'float', 'string', 'boolean']
                    "required": True
                }
            ]
        }

        form_data = {
            'name': 'Test Model',
            'version': '1.0.0',
            'domain': 'cardiology',
            'description': 'Test description',
            'input_schema': json.dumps(invalid_schema),
            'output_schema': json.dumps(self.valid_output_schema),
            'is_public': False
        }

        form = ModelUploadForm(data=form_data, files={'model_file': self.onnx_file})
        form.is_valid()

        self.assertIn('input_schema', form.errors)
        self.assertIn('invalid_type', str(form.errors['input_schema']))

    def test_input_schema_validation_min_max_range(self):
        """Test input schema validation checks min < max."""
        invalid_schema = {
            "features": [
                {
                    "name": "age",
                    "type": "integer",
                    "min": 120,
                    "max": 0,  # min > max (invalid)
                    "required": True
                }
            ]
        }

        form_data = {
            'name': 'Test Model',
            'version': '1.0.0',
            'domain': 'cardiology',
            'description': 'Test description',
            'input_schema': json.dumps(invalid_schema),
            'output_schema': json.dumps(self.valid_output_schema),
            'is_public': False
        }

        form = ModelUploadForm(data=form_data, files={'model_file': self.onnx_file})
        form.is_valid()

        self.assertIn('input_schema', form.errors)
        self.assertIn('min', str(form.errors['input_schema']).lower())

    def test_output_schema_validation_missing_type(self):
        """Test output schema validation requires 'type' field."""
        invalid_schema = {
            "classes": ["class1", "class2"]
            # Missing 'type'
        }

        form_data = {
            'name': 'Test Model',
            'version': '1.0.0',
            'domain': 'cardiology',
            'description': 'Test description',
            'input_schema': json.dumps(self.valid_input_schema),
            'output_schema': json.dumps(invalid_schema),
            'is_public': False
        }

        form = ModelUploadForm(data=form_data, files={'model_file': self.onnx_file})
        form.is_valid()

        self.assertIn('output_schema', form.errors)
        self.assertIn('type', str(form.errors['output_schema']))

    def test_output_schema_validation_invalid_type(self):
        """Test output schema validation accepts only classification/regression."""
        invalid_schema = {
            "type": "invalid_type",  # Not 'classification' or 'regression'
        }

        form_data = {
            'name': 'Test Model',
            'version': '1.0.0',
            'domain': 'cardiology',
            'description': 'Test description',
            'input_schema': json.dumps(self.valid_input_schema),
            'output_schema': json.dumps(invalid_schema),
            'is_public': False
        }

        form = ModelUploadForm(data=form_data, files={'model_file': self.onnx_file})
        form.is_valid()

        self.assertIn('output_schema', form.errors)

    def test_output_schema_classification_missing_classes(self):
        """Test classification schema requires 'classes' array."""
        invalid_schema = {
            "type": "classification"
            # Missing 'classes'
        }

        form_data = {
            'name': 'Test Model',
            'version': '1.0.0',
            'domain': 'cardiology',
            'description': 'Test description',
            'input_schema': json.dumps(self.valid_input_schema),
            'output_schema': json.dumps(invalid_schema),
            'is_public': False
        }

        form = ModelUploadForm(data=form_data, files={'model_file': self.onnx_file})
        form.is_valid()

        self.assertIn('output_schema', form.errors)
        self.assertIn('classes', str(form.errors['output_schema']))

    def test_output_schema_classification_minimum_classes(self):
        """Test classification requires at least 2 classes."""
        invalid_schema = {
            "type": "classification",
            "classes": ["single_class"]  # Only 1 class (invalid)
        }

        form_data = {
            'name': 'Test Model',
            'version': '1.0.0',
            'domain': 'cardiology',
            'description': 'Test description',
            'input_schema': json.dumps(self.valid_input_schema),
            'output_schema': json.dumps(invalid_schema),
            'is_public': False
        }

        form = ModelUploadForm(data=form_data, files={'model_file': self.onnx_file})
        form.is_valid()

        self.assertIn('output_schema', form.errors)
        self.assertIn('2', str(form.errors['output_schema']))

    def test_model_file_extension_validation(self):
        """Test that only .onnx files are accepted."""
        invalid_file = SimpleUploadedFile(
            "test_model.txt",  # Wrong extension
            b'NOT_AN_ONNX_FILE',
            content_type="text/plain"
        )

        form_data = {
            'name': 'Test Model',
            'version': '1.0.0',
            'domain': 'cardiology',
            'description': 'Test description',
            'input_schema': json.dumps(self.valid_input_schema),
            'output_schema': json.dumps(self.valid_output_schema),
            'is_public': False
        }

        form = ModelUploadForm(data=form_data, files={'model_file': invalid_file})
        form.is_valid()

        self.assertIn('model_file', form.errors)
        self.assertIn('.onnx', str(form.errors['model_file']))

    def test_form_widgets_have_bootstrap_classes(self):
        """Test that form widgets include Bootstrap CSS classes."""
        form = ModelUploadForm()

        self.assertIn('form-control', form.fields['name'].widget.attrs.get('class', ''))
        self.assertIn('form-control', form.fields['version'].widget.attrs.get('class', ''))
        self.assertIn('form-select', form.fields['domain'].widget.attrs.get('class', ''))
        self.assertIn('form-control', form.fields['description'].widget.attrs.get('class', ''))
        self.assertIn('form-control', form.fields['accuracy_percent'].widget.attrs.get('class', ''))

    def test_input_schema_accepts_csv_header_comma_separated(self):
        """Test input schema accepts comma-separated feature names."""
        form_data = {
            'name': 'Test Model',
            'version': '1.0.0',
            'domain': 'cardiology',
            'description': 'Test description',
            'input_schema': 'feature1,feature2,feature3',  # CSV header format
            'output_schema': json.dumps(self.valid_output_schema),
            'is_public': False
        }

        form = ModelUploadForm(data=form_data, files={'model_file': self.onnx_file})
        form.is_valid()

        # input_schema should be valid
        self.assertNotIn('input_schema', form.errors)

    def test_input_schema_accepts_newline_separated(self):
        """Test input schema accepts one feature per line."""
        form_data = {
            'name': 'Test Model',
            'version': '1.0.0',
            'domain': 'cardiology',
            'description': 'Test description',
            'input_schema': 'cg16057915\ncg02849695\ncg12135344',  # One per line
            'output_schema': json.dumps(self.valid_output_schema),
            'is_public': False
        }

        form = ModelUploadForm(data=form_data, files={'model_file': self.onnx_file})
        form.is_valid()

        self.assertNotIn('input_schema', form.errors)

    def test_input_schema_accepts_json_array(self):
        """Test input schema accepts JSON array of feature names."""
        form_data = {
            'name': 'Test Model',
            'version': '1.0.0',
            'domain': 'cardiology',
            'description': 'Test description',
            'input_schema': '["feature1", "feature2", "feature3"]',  # JSON array
            'output_schema': json.dumps(self.valid_output_schema),
            'is_public': False
        }

        form = ModelUploadForm(data=form_data, files={'model_file': self.onnx_file})
        form.is_valid()

        self.assertNotIn('input_schema', form.errors)

    def test_input_schema_skips_id_column(self):
        """Test that 'id' column is automatically skipped."""
        form_data = {
            'name': 'Test Model',
            'version': '1.0.0',
            'domain': 'cardiology',
            'description': 'Test description',
            'input_schema': 'id\nfeature1\nfeature2',  # With 'id' that should be skipped
            'output_schema': json.dumps(self.valid_output_schema),
            'is_public': False
        }

        form = ModelUploadForm(data=form_data, files={'model_file': self.onnx_file})
        form.is_valid()

        self.assertNotIn('input_schema', form.errors)

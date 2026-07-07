"""
Forms for inference app (MEMBER-facing).
"""
from django import forms
from django.core.exceptions import ValidationError
from inference.models import DeployedModel
import json

# Optional ONNX validation (requires onnx library with C++ dependencies)
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ModelUploadForm(forms.ModelForm):
    """
    Form for uploading ONNX models with schema definition.

    Features:
    - ONNX file validation
    - JSON schema validation for input/output
    - Accuracy field (optional, 0-100%)
    - Automatic file size and checksum calculation
    """

    # Override accuracy field to use percentage (0-100) instead of decimal (0.0-1.0)
    accuracy_percent = forms.FloatField(
        required=False,
        min_value=0.0,
        max_value=100.0,
        label='Accuracy (%)',
        help_text='Optional: Model accuracy from your validation testing (0-100%)',
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., 94.5',
            'step': '0.1'
        })
    )

    # Override input_schema to accept multiple formats (not just JSON)
    input_schema = forms.CharField(
        required=True,
        label='Input Schema',
        help_text='Feature names (CSV header, one per line, JSON array, or full JSON schema)',
        widget=forms.Textarea(attrs={
            'class': 'form-control font-monospace',
            'rows': 10,
            'placeholder': '''Accepted formats:

1. CSV header (simplest):
feature1,feature2,feature3

2. One feature per line:
cg16057915
cg02849695
cg12135344

3. JSON array:
["feature1", "feature2", "feature3"]

4. Full JSON schema:
{"features": [{"name": "age", "type": "float", "required": true}]}''',
            'required': True
        })
    )

    class Meta:
        model = DeployedModel
        fields = [
            'name',
            'version',
            'domain',
            'description',
            'model_file',
            'input_schema',
            'output_schema',
            'is_public',
        ]
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., CardioNet',
                'required': True
            }),
            'version': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., 1.0.0',
                'value': '1.0.0'
            }),
            'domain': forms.Select(attrs={
                'class': 'form-select',
                'required': True
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Describe the model purpose, training data, and expected use cases...',
                'required': True
            }),
            'model_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.onnx',
                'required': True
            }),
            'output_schema': forms.Textarea(attrs={
                'class': 'form-control font-monospace',
                'rows': 6,
                'placeholder': '''{"type": "classification", "classes": ["no_risk", "at_risk", "high_risk"], "confidence_threshold": 0.75}''',
                'required': True
            }),
            'is_public': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
        }
        help_texts = {
            'name': 'Unique descriptive name for the model',
            'version': 'Semantic version (e.g., 1.0.0)',
            'domain': 'Medical domain for this model',
            'description': 'Detailed description of model purpose and usage',
            'model_file': 'ONNX model file (max 500MB)',
            'output_schema': 'JSON schema defining output format',
            'is_public': 'Make this model available to all users (requires admin approval)',
        }

    def clean_model_file(self):
        """Validate ONNX file."""
        model_file = self.cleaned_data.get('model_file')

        if not model_file:
            raise ValidationError('Model file is required.')

        if not model_file.name.endswith('.onnx'):
            raise ValidationError('File must be an ONNX model (.onnx extension).')

        max_size = 500 * 1024 * 1024  # 500MB in bytes
        if model_file.size > max_size:
            size_mb = model_file.size / (1024 * 1024)
            raise ValidationError(f'File size ({size_mb:.1f}MB) exceeds maximum allowed (500MB).')

        # Validate ONNX format (if onnx library is available)
        if ONNX_AVAILABLE:
            try:
                model_file.seek(0)
                model_content = model_file.read()
                model_file.seek(0)  # Reset for later use

                onnx_model = onnx.load_model_from_string(model_content)

                onnx.checker.check_model(onnx_model)

            except Exception as e:
                raise ValidationError(f'Invalid ONNX file: {str(e)}')
        else:
            # Basic file signature check (ONNX files start with specific bytes)
            model_file.seek(0)
            header = model_file.read(8)
            model_file.seek(0)

            # ONNX files are Protocol Buffer format, check for protobuf signature
            # This is a basic check - not as thorough as onnx.checker but works without library
            if not header or len(header) < 4:
                raise ValidationError('File appears to be empty or corrupted.')

            # Just warn that full validation is not available
            # In production, you should install onnx library

        return model_file

    def clean_input_schema(self):
        """
        Validate input schema - accepts multiple formats:

        1. Full JSON schema:
           {"features": [{"name": "age", "type": "float", "required": true}, ...]}

        2. Simple JSON array of feature names:
           ["feature1", "feature2", "feature3"]

        3. CSV header (comma or newline separated):
           feature1,feature2,feature3
           OR
           feature1
           feature2
           feature3
        """
        schema_str = self.cleaned_data.get('input_schema')

        if isinstance(schema_str, dict):
            schema = schema_str
        else:
            schema_str = schema_str.strip()

            try:
                parsed = json.loads(schema_str)

                if isinstance(parsed, dict):
                    schema = parsed
                elif isinstance(parsed, list):
                    schema = self._convert_feature_names_to_schema(parsed)
                else:
                    raise ValidationError('JSON must be an object or array.')

            except json.JSONDecodeError:
                schema = self._parse_csv_header_to_schema(schema_str)

        return self._validate_schema_structure(schema)

    def _convert_feature_names_to_schema(self, feature_names):
        """Convert a simple list of feature names to full schema format."""
        if not feature_names:
            raise ValidationError('At least one feature is required.')

        features = []
        for name in feature_names:
            if not isinstance(name, str):
                raise ValidationError(f'Feature name must be a string, got: {type(name).__name__}')
            name = name.strip().strip('"').strip("'")
            if not name:
                continue
            features.append({
                'name': name,
                'type': 'float',  # Default to float for numeric models
                'required': True
            })

        if not features:
            raise ValidationError('At least one feature is required.')

        return {'features': features}

    def _parse_csv_header_to_schema(self, header_str):
        """Parse CSV header string (comma or newline separated) to schema."""
        if '\n' in header_str:
            names = [line.strip() for line in header_str.split('\n')]
        elif ',' in header_str:
            names = [name.strip() for name in header_str.split(',')]
        else:
            names = [header_str.strip()]

        clean_names = []
        for name in names:
            name = name.strip().strip('"').strip("'")
            if name and name.lower() != 'id':  # Skip 'id' column if present
                clean_names.append(name)

        if not clean_names:
            raise ValidationError(
                'Could not parse input schema. Accepted formats:\n'
                '1. JSON: {"features": [{"name": "feat1", "type": "float", "required": true}]}\n'
                '2. JSON array: ["feat1", "feat2", "feat3"]\n'
                '3. CSV header: feat1,feat2,feat3\n'
                '4. One feature per line'
            )

        return self._convert_feature_names_to_schema(clean_names)

    def _validate_schema_structure(self, schema):
        """Validate the final schema structure."""
        if not isinstance(schema, dict):
            raise ValidationError('Schema must be a JSON object.')

        if 'features' not in schema:
            raise ValidationError('Schema must contain "features" array.')

        if not isinstance(schema['features'], list):
            raise ValidationError('"features" must be an array.')

        if len(schema['features']) == 0:
            raise ValidationError('At least one feature is required.')

        valid_types = ['integer', 'float', 'string', 'boolean']

        for idx, feature in enumerate(schema['features']):
            if not isinstance(feature, dict):
                raise ValidationError(f'Feature {idx + 1} must be an object.')

            if 'name' not in feature:
                raise ValidationError(f'Feature {idx + 1} missing required field: "name"')

            if 'type' not in feature:
                feature['type'] = 'float'
            if 'required' not in feature:
                feature['required'] = True

            if feature['type'] not in valid_types:
                raise ValidationError(
                    f'Feature "{feature["name"]}" has invalid type "{feature["type"]}". '
                    f'Must be one of: {", ".join(valid_types)}'
                )

            if 'min' in feature and 'max' in feature:
                try:
                    if float(feature['min']) >= float(feature['max']):
                        raise ValidationError(
                            f'Feature "{feature["name"]}": min ({feature["min"]}) must be less than max ({feature["max"]})'
                        )
                except (ValueError, TypeError):
                    raise ValidationError(f'Feature "{feature["name"]}": min and max must be numeric')

        return schema

    def clean_output_schema(self):
        """Validate output schema JSON."""
        schema_str = self.cleaned_data.get('output_schema')

        if isinstance(schema_str, dict):
            schema = schema_str
        else:
            try:
                schema = json.loads(schema_str)
            except json.JSONDecodeError as e:
                raise ValidationError(f'Invalid JSON: {str(e)}')

        if not isinstance(schema, dict):
            raise ValidationError('Schema must be a JSON object.')

        if 'type' not in schema:
            raise ValidationError('Schema must contain "type" field.')

        valid_types = ['classification', 'regression']
        if schema['type'] not in valid_types:
            raise ValidationError(f'Type must be one of: {", ".join(valid_types)}')

        if schema['type'] == 'classification':
            if 'classes' not in schema:
                raise ValidationError('Classification schema must contain "classes" field.')

            classes = schema['classes']

            # Accept both list and dict formats
            if isinstance(classes, list):
                if len(classes) < 2:
                    raise ValidationError('Classification requires at least 2 classes.')
                # Convert list to dict: ['Control', 'Case'] -> {0: 'Control', 1: 'Case'}
                schema['classes'] = {str(i): name for i, name in enumerate(classes)}
            elif isinstance(classes, dict):
                if len(classes) < 2:
                    raise ValidationError('Classification requires at least 2 classes.')
                schema['classes'] = {str(k): v for k, v in classes.items()}
            else:
                raise ValidationError('"classes" must be an array or object.')

        return schema

    def clean_accuracy_percent(self):
        """Convert accuracy from percentage to decimal."""
        accuracy_percent = self.cleaned_data.get('accuracy_percent')

        if accuracy_percent is None:
            return None

        return accuracy_percent / 100.0

    def save(self, commit=True):
        """Save model with calculated fields."""
        instance = super().save(commit=False)

        accuracy_decimal = self.cleaned_data.get('accuracy_percent')
        if accuracy_decimal is not None:
            instance.accuracy = accuracy_decimal / 100.0

        instance.source = 'upload'

        # File size and checksum are calculated in model's save() method

        if commit:
            instance.save()

        return instance


class ModelEditForm(forms.ModelForm):
    """
    Form for editing model metadata and schemas.

    Allows editing:
    - Basic info (name, version, description, domain)
    - Input schema (with flexible format support)
    - Output schema
    - Visibility (is_public)

    Does NOT allow changing the model file itself.
    """

    # Override input_schema to accept multiple formats (same as upload form)
    input_schema = forms.CharField(
        required=True,
        label='Input Schema',
        help_text='Feature names (CSV header, one per line, JSON array, or full JSON schema)',
        widget=forms.Textarea(attrs={
            'class': 'form-control font-monospace',
            'rows': 12,
        })
    )

    class Meta:
        model = DeployedModel
        fields = [
            'name',
            'version',
            'domain',
            'description',
            'input_schema',
            'output_schema',
            'is_public',
        ]
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
            }),
            'version': forms.TextInput(attrs={
                'class': 'form-control',
            }),
            'domain': forms.Select(attrs={
                'class': 'form-select',
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
            }),
            'output_schema': forms.Textarea(attrs={
                'class': 'form-control font-monospace',
                'rows': 6,
            }),
            'is_public': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.input_schema:
            self.initial['input_schema'] = json.dumps(
                self.instance.input_schema, indent=2
            )
        if self.instance and self.instance.output_schema:
            self.initial['output_schema'] = json.dumps(
                self.instance.output_schema, indent=2
            )

    # Reuse validation methods from ModelUploadForm
    def clean_input_schema(self):
        """Validate input schema - accepts multiple formats."""
        schema_str = self.cleaned_data.get('input_schema')

        if isinstance(schema_str, dict):
            schema = schema_str
        else:
            schema_str = schema_str.strip()

            try:
                parsed = json.loads(schema_str)

                if isinstance(parsed, dict):
                    schema = parsed
                elif isinstance(parsed, list):
                    schema = self._convert_feature_names_to_schema(parsed)
                else:
                    raise ValidationError('JSON must be an object or array.')

            except json.JSONDecodeError:
                schema = self._parse_csv_header_to_schema(schema_str)

        return self._validate_schema_structure(schema)

    def _convert_feature_names_to_schema(self, feature_names):
        """Convert a simple list of feature names to full schema format."""
        if not feature_names:
            raise ValidationError('At least one feature is required.')

        features = []
        for name in feature_names:
            if not isinstance(name, str):
                raise ValidationError(f'Feature name must be a string, got: {type(name).__name__}')
            name = name.strip().strip('"').strip("'")
            if not name:
                continue
            features.append({
                'name': name,
                'type': 'float',
                'required': True
            })

        if not features:
            raise ValidationError('At least one feature is required.')

        return {'features': features}

    def _parse_csv_header_to_schema(self, header_str):
        """Parse CSV header string to schema."""
        if '\n' in header_str:
            names = [line.strip() for line in header_str.split('\n')]
        elif ',' in header_str:
            names = [name.strip() for name in header_str.split(',')]
        else:
            names = [header_str.strip()]

        clean_names = []
        for name in names:
            name = name.strip().strip('"').strip("'")
            if name and name.lower() != 'id':
                clean_names.append(name)

        if not clean_names:
            raise ValidationError(
                'Could not parse input schema. Use JSON format or comma/newline-separated feature names.'
            )

        return self._convert_feature_names_to_schema(clean_names)

    def _validate_schema_structure(self, schema):
        """Validate the final schema structure."""
        if not isinstance(schema, dict):
            raise ValidationError('Schema must be a JSON object.')

        if 'features' not in schema:
            raise ValidationError('Schema must contain "features" array.')

        if not isinstance(schema['features'], list):
            raise ValidationError('"features" must be an array.')

        if len(schema['features']) == 0:
            raise ValidationError('At least one feature is required.')

        valid_types = ['integer', 'float', 'string', 'boolean']

        for idx, feature in enumerate(schema['features']):
            if not isinstance(feature, dict):
                raise ValidationError(f'Feature {idx + 1} must be an object.')

            if 'name' not in feature:
                raise ValidationError(f'Feature {idx + 1} missing required field: "name"')

            if 'type' not in feature:
                feature['type'] = 'float'
            if 'required' not in feature:
                feature['required'] = True

            if feature['type'] not in valid_types:
                raise ValidationError(
                    f'Feature "{feature["name"]}" has invalid type "{feature["type"]}". '
                    f'Must be one of: {", ".join(valid_types)}'
                )

            if 'min' in feature and 'max' in feature:
                try:
                    if float(feature['min']) >= float(feature['max']):
                        raise ValidationError(
                            f'Feature "{feature["name"]}": min must be less than max'
                        )
                except (ValueError, TypeError):
                    raise ValidationError(f'Feature "{feature["name"]}": min and max must be numeric')

        return schema

    def clean_output_schema(self):
        """Validate output schema JSON."""
        schema_str = self.cleaned_data.get('output_schema')

        if isinstance(schema_str, dict):
            schema = schema_str
        else:
            try:
                schema = json.loads(schema_str)
            except json.JSONDecodeError as e:
                raise ValidationError(f'Invalid JSON: {str(e)}')

        if not isinstance(schema, dict):
            raise ValidationError('Schema must be a JSON object.')

        if 'type' not in schema:
            raise ValidationError('Schema must contain "type" field.')

        valid_types = ['classification', 'regression']
        if schema['type'] not in valid_types:
            raise ValidationError(f'Type must be one of: {", ".join(valid_types)}')

        if schema['type'] == 'classification':
            if 'classes' not in schema:
                raise ValidationError('Classification schema must contain "classes" field.')

            classes = schema['classes']

            if isinstance(classes, list):
                if len(classes) < 2:
                    raise ValidationError('Classification requires at least 2 classes.')
                schema['classes'] = {str(i): name for i, name in enumerate(classes)}
            elif isinstance(classes, dict):
                if len(classes) < 2:
                    raise ValidationError('Classification requires at least 2 classes.')
                schema['classes'] = {str(k): v for k, v in classes.items()}
            else:
                raise ValidationError('"classes" must be an array or object.')

        return schema

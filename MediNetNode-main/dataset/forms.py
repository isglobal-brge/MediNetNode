"""
Forms for dataset management.
"""

from django import forms
from .models import Dataset

class DatasetUploadForm(forms.ModelForm):
    """Form for uploading new datasets."""
    
    file = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.csv,.json,.parquet,.h5,.npy'
        }),
        help_text='Allowed formats: CSV, JSON, Parquet, HDF5, NumPy'
    )
    
    class Meta:
        model = Dataset
        fields = ['name', 'description', 'medical_domain', 'data_type']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter dataset name'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Describe the dataset content and purpose'
            }),
            'medical_domain': forms.Select(attrs={
                'class': 'form-control'
            }),
            'data_type': forms.Select(attrs={
                'class': 'form-control'
            })
        }
    
    def clean_name(self):
        """Validate dataset name."""
        name = self.cleaned_data.get('name')
        
        if not name:
            raise forms.ValidationError("Dataset name is required.")
        
        # Check for existing dataset with same name
        if Dataset.objects.using('datasets_db').filter(name=name, is_active=True).exists():
            raise forms.ValidationError("A dataset with this name already exists.")
        
        return name
    
    def clean_file(self):
        """Validate uploaded file."""
        file = self.cleaned_data.get('file')
        
        if not file:
            raise forms.ValidationError("File is required.")
        
        # No size limit for medical datasets (they can be very large)
        # Only check for empty files
        if file.size == 0:
            raise forms.ValidationError("File cannot be empty.")
        
        # Check file extension
        allowed_extensions = ['.csv', '.json', '.parquet', '.h5', '.npy']
        file_extension = None
        
        if hasattr(file, 'name') and file.name:
            import os
            file_extension = os.path.splitext(file.name)[1].lower()
        
        if not file_extension or file_extension not in allowed_extensions:
            raise forms.ValidationError(
                f"File type not allowed. Allowed extensions: {', '.join(allowed_extensions)}"
            )
        
        return file


from .models import Dataset, DatasetMetadata

class DatasetMetadataForm(forms.ModelForm):
    """Form for editing dataset metadata."""
    
    class Meta:
        model = DatasetMetadata
        fields = [
            'statistical_summary',
            'missing_values',
            'data_distribution',
            'quality_score',
            'completeness_percentage'
        ]
        widgets = {
            'statistical_summary': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'missing_values': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'data_distribution': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'quality_score': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '0',
                'max': '1',
                'step': '0.01'
            }),
            'completeness_percentage': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '0',
                'max': '100',
                'step': '0.01'
            })
        }
    
    def clean_column_info(self):
        """Validate column_info JSON structure."""
        data = self.cleaned_data['column_info']
        if not isinstance(data, dict):
            raise forms.ValidationError("Column info must be a valid JSON object")
        return data
    
    def clean_data_types(self):
        """Validate data_types JSON structure."""
        data = self.cleaned_data['data_types']
        if not isinstance(data, dict):
            raise forms.ValidationError("Data types must be a valid JSON object")
        return data
    
    def clean_missing_values(self):
        """Validate missing_values JSON structure."""
        data = self.cleaned_data['missing_values']
        if not isinstance(data, dict):
            raise forms.ValidationError("Missing values must be a valid JSON object")
        return data
    
    def clean_value_ranges(self):
        """Validate value_ranges JSON structure."""
        data = self.cleaned_data['value_ranges']
        if not isinstance(data, dict):
            raise forms.ValidationError("Value ranges must be a valid JSON object")
        return data
    
    def clean_categorical_mappings(self):
        """Validate categorical_mappings JSON structure."""
        data = self.cleaned_data['categorical_mappings']
        if not isinstance(data, dict):
            raise forms.ValidationError("Categorical mappings must be a valid JSON object")
        return data


class DatasetEditForm(forms.ModelForm):
    """Form for editing existing dataset information."""

    target_column = forms.ChoiceField(
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-select',
            'id': 'target_column_select'
        }),
        help_text='Column name used for predictions/outcomes'
    )

    class Meta:
        model = Dataset
        fields = [
            'name',
            'description',
            'medical_domain',
            'anonymized',
            'target_column'
        ]
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter dataset name'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Describe the dataset content and purpose'
            }),
            'medical_domain': forms.Select(attrs={
                'class': 'form-select'
            }),
            'anonymized': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            })
        }
        help_texts = {
            'name': 'Unique identifier for the dataset',
            'medical_domain': 'Medical specialty or area',
            'anonymized': 'Check if dataset has been properly anonymized'
        }

    def __init__(self, *args, **kwargs):
        """Initialize form and populate target_column choices from dataset file."""
        super().__init__(*args, **kwargs)

        # Get column choices from the dataset file
        if self.instance and self.instance.pk:
            try:
                from .uploader import SecureDatasetUploader

                # Create uploader instance (user doesn't matter for column extraction)
                uploader = SecureDatasetUploader(user=None)

                # Get columns from file
                file_path = self.instance.file_path
                columns = uploader.get_csv_columns(file_path)

                # Create choices: (value, display_label)
                choices = [('', '--- Select Target Column ---')]
                choices.extend([(col, col) for col in columns])

                self.fields['target_column'].choices = choices

            except Exception as e:
                # If we can't read columns, provide empty choice
                self.fields['target_column'].choices = [
                    ('', '--- Unable to read columns ---')
                ]
                self.fields['target_column'].help_text = f'Error reading columns: {str(e)}'

    def clean_name(self):
        """Validate dataset name - allow same name for editing."""
        name = self.cleaned_data.get('name')

        if not name:
            raise forms.ValidationError("Dataset name is required.")

        # Check for existing dataset with same name, excluding current instance
        existing = Dataset.objects.using('datasets_db').filter(
            name=name,
            is_active=True
        ).exclude(pk=self.instance.pk if self.instance else None)

        if existing.exists():
            raise forms.ValidationError("A dataset with this name already exists.")

        return name


class DatasetFilterForm(forms.Form):
    """Form for filtering datasets."""
    
    search = forms.CharField(
        max_length=200,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search datasets...'
        })
    )
    
    medical_domain = forms.ChoiceField(
        choices=[('', 'All Domains')] + list(Dataset.MEDICAL_DOMAINS),
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )
    
    data_type = forms.ChoiceField(
        choices=[('', 'All Types')] + list(Dataset.DATA_TYPES),
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )
    
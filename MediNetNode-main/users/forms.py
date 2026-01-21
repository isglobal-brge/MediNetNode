from django import forms
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from .models import CustomUser, Role


class SecureUserCreationForm(UserCreationForm):
    """Secure user creation form with enhanced validation."""
    
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'user@example.com'
        })
    )
    
    first_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'First name'
        })
    )
    
    last_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Last name'
        })
    )
    
    role = forms.ModelChoiceField(
        queryset=Role.objects.all(),
        required=True,
        empty_label="Select role",
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    class Meta:
        model = CustomUser
        fields = ('username', 'email', 'first_name', 'last_name', 'role', 'password1', 'password2')
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'username'
            }),
        }

    def __init__(self, *args, **kwargs):
        self.created_by = kwargs.pop('created_by', None)
        super().__init__(*args, **kwargs)
        
        # Add Bootstrap classes to password fields
        self.fields['password1'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Password'
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Confirm password'
        })

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if CustomUser.objects.filter(email=email).exists():
            raise ValidationError("A user with this email already exists.")
        return email

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if len(username) < 3:
            raise ValidationError("Username must have at least 3 characters.")
        if not username.replace('_', '').isalnum():
            raise ValidationError("Username may contain letters, numbers and underscores only.")
        return username

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        user.role = self.cleaned_data['role']
        
        if self.created_by:
            user.created_by = self.created_by
            
        if commit:
            user.save()
        return user


class UserUpdateForm(UserChangeForm):
    """Form for updating user information."""
    
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control'})
    )
    
    first_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    last_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    role = forms.ModelChoiceField(
        queryset=Role.objects.all(),
        required=True,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    is_active = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    class Meta:
        model = CustomUser
        fields = ('username', 'email', 'first_name', 'last_name', 'role', 'is_active')
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control', 'readonly': True}),
        }

    def __init__(self, *args, **kwargs):
        self.request_user = kwargs.pop('request_user', None)
        super().__init__(*args, **kwargs)
        
        # Remove password field
        if 'password' in self.fields:
            del self.fields['password']

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if self.instance and self.instance.email != email:
            if CustomUser.objects.filter(email=email).exists():
                raise ValidationError("A user with this email already exists.")
        return email

    def clean_role(self):
        role = self.cleaned_data.get('role')
        
        # Prevent user from changing their own role
        if (self.request_user and 
            self.instance and 
            self.request_user.id == self.instance.id):
            raise ValidationError("You cannot change your own role.")
            
        return role

    def clean_is_active(self):
        is_active = self.cleaned_data.get('is_active')
        
        # Prevent user from deactivating themselves
        if (self.request_user and 
            self.instance and 
            self.request_user.id == self.instance.id and 
            not is_active):
            raise ValidationError("You cannot deactivate your own account.")
            
        return is_active


class UserPasswordChangeForm(forms.Form):
    """Form for changing user password."""
    
    new_password1 = forms.CharField(
        label="New password",
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'New password'
        })
    )
    
    new_password2 = forms.CharField(
        label="Confirm new password",
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Confirm password'
        })
    )

    def __init__(self, user, *args, **kwargs):
        self.user = user
        super().__init__(*args, **kwargs)

    def clean_new_password1(self):
        password = self.cleaned_data.get('new_password1')
        validate_password(password, self.user)
        return password

    def clean_new_password2(self):
        password1 = self.cleaned_data.get('new_password1')
        password2 = self.cleaned_data.get('new_password2')
        
        if password1 and password2:
            if password1 != password2:
                raise ValidationError("Passwords do not match.")
        return password2

    def save(self):
        password = self.cleaned_data['new_password1']
        self.user.set_password(password)
        self.user.save()
        return self.user


class UserSearchForm(forms.Form):
    """Form for searching and filtering users."""
    
    q = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search by username, email, name...'
        })
    )
    
    role = forms.ModelChoiceField(
        queryset=Role.objects.all(),
        required=False,
        empty_label="All roles",
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    is_active = forms.ChoiceField(
        choices=[
            ('', 'All'),
            ('True', 'Active'),
            ('False', 'Inactive'),
        ],
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    ordering = forms.ChoiceField(
        choices=[
            ('username', 'Username A-Z'),
            ('-username', 'Username Z-A'),
            ('email', 'Email A-Z'),
            ('-email', 'Email Z-A'),
            ('date_joined', 'Created (oldest)'),
            ('-date_joined', 'Created (newest)'),
            ('last_login', 'Last login (oldest)'),
            ('-last_login', 'Last login (newest)'),
        ],
        required=False,
        initial='-date_joined',
        widget=forms.Select(attrs={'class': 'form-select'})
    )
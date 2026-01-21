import tempfile
import os
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from users.models import CustomUser, Role
from dataset.models import Dataset
from audit.models import AuditEvent
import pandas as pd


User = get_user_model()


class RealDataAnalysisAccessControlTest(TestCase):
    """Test access control for real data analysis - AUDITOR ONLY."""
    
    databases = ['default', 'datasets_db']
    
    def setUp(self):
        """Set up test users with different roles."""
        # Create roles
        self.admin_role = Role.objects.get(name='ADMIN')
        self.investigador_role = Role.objects.get(name='RESEARCHER')
        self.auditor_role = Role.objects.get(name='AUDITOR')
        
        # Create users with different roles
        self.admin_user = CustomUser.objects.create_user(
            username='admin_user',
            password='StrongPass123!',
            role=self.admin_role
        )
        
        self.investigador_user = CustomUser.objects.create_user(
            username='investigador_user', 
            password='StrongPass123!',
            role=self.investigador_role
        )
        
        self.auditor_user = CustomUser.objects.create_user(
            username='auditor_user',
            password='StrongPass123!', 
            role=self.auditor_role
        )
        
        self.client = Client()

    def test_real_data_analysis_access_auditor_allowed(self):
        """Test that AUDITOR role can access real data analysis."""
        self.client.login(username='auditor_user', password='StrongPass123!')
        response = self.client.get(reverse('audit:medical_data_analysis'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'MEDICAL DATASET ANALYSIS')
        self.assertContains(response, 'AUDITOR ACCESS ONLY')

    def test_real_data_analysis_access_admin_denied(self):
        """Test that ADMIN role is denied access to real data analysis."""
        self.client.login(username='admin_user', password='StrongPass123!')
        response = self.client.get(reverse('audit:medical_data_analysis'))
        self.assertEqual(response.status_code, 403)

    def test_real_data_analysis_access_investigador_denied(self):
        """Test that INVESTIGADOR role is denied access to real data analysis."""
        self.client.login(username='investigador_user', password='StrongPass123!')
        response = self.client.get(reverse('audit:medical_data_analysis'))
        # RESEARCHER gets redirected to info page (web access blocked)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/info/researcher/', response.url)

    def test_real_data_analysis_access_anonymous_denied(self):
        """Test that anonymous users are denied access to real data analysis."""
        response = self.client.get(reverse('audit:medical_data_analysis'))
        # Should redirect to login page
        self.assertEqual(response.status_code, 302)
        self.assertIn('/auth/login/', response.url)

    def test_access_logging_for_real_data_request(self):
        """Test that access to real data analysis page is logged."""
        self.client.login(username='auditor_user', password='StrongPass123!')
        
        # Clear existing audit events
        AuditEvent.objects.all().delete()
        
        response = self.client.get(reverse('audit:medical_data_analysis'))
        self.assertEqual(response.status_code, 200)
        
        # Check that access was logged
        audit_events = AuditEvent.objects.filter(
            action='REAL_DATA_ACCESS_REQUEST',
            user=self.auditor_user
        )
        self.assertEqual(audit_events.count(), 1)
        
        audit_event = audit_events.first()
        self.assertEqual(audit_event.resource, 'dataset_real_data_analysis')
        self.assertTrue(audit_event.success)
        self.assertIn('watermark_applied', audit_event.details)
        self.assertEqual(audit_event.details['access_level'], 'AUDITOR_REAL_DATA')


class RealDataAnalysisFunctionalityTest(TestCase):
    """Test functionality of real data analysis for authorized users."""
    
    databases = ['default', 'datasets_db']
    
    def setUp(self):
        """Set up test environment with sample data."""
        self.auditor_role = Role.objects.get(name='AUDITOR')
        
        self.auditor_user = CustomUser.objects.create_user(
            username='auditor_user',
            password='StrongPass123!',
            role=self.auditor_role
        )
        
        # Create a temporary CSV file for testing
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        test_data = """age,gender,condition,treatment,outcome
25,M,hypertension,medication_a,improved
30,F,diabetes,insulin,stable
35,M,heart_disease,surgery,recovered
40,F,cancer,chemotherapy,stable
28,M,depression,therapy,improved"""
        self.temp_file.write(test_data)
        self.temp_file.close()
        
        # Create a test dataset
        self.test_dataset = Dataset.objects.using('datasets_db').create(
            name='Test Medical Dataset',
            description='Test dataset for auditor analysis',
            file_path=self.temp_file.name,
            file_size=len(test_data),
            medical_domain='cardiology',
            patient_count=5,
            data_type='tabular',
            uploaded_by_id=self.auditor_user.id
        )
        
        self.client = Client()
        self.client.login(username='auditor_user', password='StrongPass123!')

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_dataset_selection_displays_available_datasets(self):
        """Test that available datasets are displayed for selection."""
        response = self.client.get(reverse('audit:medical_data_analysis'))
        self.assertEqual(response.status_code, 200)
        
        self.assertContains(response, 'Test Medical Dataset')
        self.assertContains(response, 'cardiology')
        self.assertContains(response, 'Choose a dataset to analyze')

    def test_dataset_data_preview_limited_to_100_rows(self):
        """Test that dataset preview is limited to 100 rows maximum."""
        response = self.client.get(
            reverse('audit:medical_data_analysis'),
            {'dataset_id': self.test_dataset.id}
        )
        self.assertEqual(response.status_code, 200)
        
        # Check that watermarking is applied
        self.assertContains(response, 'CONFIDENTIAL MEDICAL DATA')
        self.assertContains(response, 'AUTHORIZED ACCESS ONLY')
        
        # Check data is displayed
        self.assertContains(response, 'REAL MEDICAL DATA PREVIEW')
        self.assertContains(response, 'LIMITED TO 100 ROWS')
        
        # Check actual data content
        self.assertContains(response, 'hypertension')
        self.assertContains(response, 'diabetes')
        self.assertContains(response, 'medication_a')

    def test_watermark_information_displayed(self):
        """Test that watermark information is properly displayed."""
        response = self.client.get(
            reverse('audit:medical_data_analysis'),
            {'dataset_id': self.test_dataset.id}
        )
        self.assertEqual(response.status_code, 200)
        
        # Check watermark elements
        self.assertContains(response, 'watermark-info')
        self.assertContains(response, 'watermark-bottom')
        self.assertContains(response, self.auditor_user.username.upper())
        self.assertContains(response, 'HASH:')

    def test_anonymization_analysis_performed(self):
        """Test that anonymization quality analysis is performed."""
        response = self.client.get(
            reverse('audit:medical_data_analysis'),
            {'dataset_id': self.test_dataset.id}
        )
        self.assertEqual(response.status_code, 200)
        
        # Check anonymization analysis section
        self.assertContains(response, 'Anonymization Quality')
        self.assertContains(response, '/100')  # Score display

    def test_compliance_analysis_performed(self):
        """Test that medical compliance analysis is performed."""
        response = self.client.get(
            reverse('audit:medical_data_analysis'),
            {'dataset_id': self.test_dataset.id}
        )
        self.assertEqual(response.status_code, 200)
        
        # Check compliance analysis section
        self.assertContains(response, 'Medical Compliance Analysis')
        self.assertContains(response, 'K-Anonymity')
        self.assertContains(response, 'Data Completeness')

    def test_column_analysis_displayed(self):
        """Test that column analysis is displayed correctly."""
        response = self.client.get(
            reverse('audit:medical_data_analysis'),
            {'dataset_id': self.test_dataset.id}
        )
        self.assertEqual(response.status_code, 200)
        
        # Check column analysis
        self.assertContains(response, 'Column Analysis')
        self.assertContains(response, 'age')
        self.assertContains(response, 'gender')
        self.assertContains(response, 'condition')
        self.assertContains(response, 'Type')
        self.assertContains(response, 'Missing')
        self.assertContains(response, 'Unique')

    def test_real_data_access_logged_with_details(self):
        """Test that real data access is logged with full details."""
        # Clear existing audit events
        AuditEvent.objects.all().delete()
        
        response = self.client.get(
            reverse('audit:medical_data_analysis'),
            {'dataset_id': self.test_dataset.id}
        )
        self.assertEqual(response.status_code, 200)
        
        # Check that detailed access was logged
        preview_events = AuditEvent.objects.filter(
            action='REAL_DATA_PREVIEW_ACCESSED',
            user=self.auditor_user
        )
        self.assertEqual(preview_events.count(), 1)
        
        preview_event = preview_events.first()
        self.assertEqual(preview_event.resource, f'dataset:{self.test_dataset.name}')
        self.assertTrue(preview_event.success)
        
        # Check detailed logging
        details = preview_event.details
        self.assertIn('dataset_id', details)
        self.assertIn('rows_accessed', details)
        self.assertIn('columns_accessed', details)
        self.assertIn('watermark_hash', details)
        self.assertIn('medical_domain', details)
        self.assertIn('anonymization_score', details)

    def test_nonexistent_dataset_error_handling(self):
        """Test error handling for nonexistent dataset."""
        response = self.client.get(
            reverse('audit:medical_data_analysis'),
            {'dataset_id': 99999}  # Non-existent ID
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Dataset not found')

    def test_file_not_found_error_handling(self):
        """Test error handling when dataset file is not found."""
        # Create dataset with invalid file path
        invalid_dataset = Dataset.objects.using('datasets_db').create(
            name='Invalid Dataset',
            description='Dataset with invalid file path',
            file_path='/nonexistent/path/file.csv',
            file_size=100,
            medical_domain='test',
            patient_count=10,
            data_type='tabular',
            uploaded_by_id=self.auditor_user.id
        )
        
        response = self.client.get(
            reverse('audit:medical_data_analysis'),
            {'dataset_id': invalid_dataset.id}
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Dataset file not found on filesystem')


class AnonymizationAnalysisTest(TestCase):
    """Test anonymization quality analysis functionality."""
    
    databases = ['default', 'datasets_db']
    
    def setUp(self):
        """Set up test environment."""
        self.auditor_role = Role.objects.get(name='AUDITOR')
        self.auditor_user = CustomUser.objects.create_user(
            username='auditor_user',
            password='StrongPass123!',
            role=self.auditor_role
        )
        self.client = Client()
        self.client.login(username='auditor_user', password='StrongPass123!')

    def test_anonymization_analysis_with_identifiers(self):
        """Test anonymization analysis detects potential identifiers."""
        # Create CSV with potential identifier columns
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        test_data = """patient_name,email,phone,age,condition
John Doe,john@email.com,555-1234,30,diabetes
Jane Smith,jane@email.com,555-5678,25,hypertension"""
        temp_file.write(test_data)
        temp_file.close()
        
        try:
            dataset = Dataset.objects.using('datasets_db').create(
                name='Dataset with Identifiers',
                description='Test dataset with potential identifiers',
                file_path=temp_file.name,
                file_size=len(test_data),
                medical_domain='general',
                patient_count=2,
                data_type='tabular',
                uploaded_by_id=self.auditor_user.id
            )
            
            response = self.client.get(
                reverse('audit:medical_data_analysis'),
                {'dataset_id': dataset.id}
            )
            self.assertEqual(response.status_code, 200)
            
            # Should detect issues with anonymization
            self.assertContains(response, 'Anonymization Quality')
            # The score should be lower due to identifier columns
            
        finally:
            os.unlink(temp_file.name)

    def test_compliance_analysis_scoring(self):
        """Test medical compliance analysis scoring."""
        # Create CSV that should pass most compliance checks
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        test_data = """age,gender,condition,treatment,outcome
""" + "\n".join([f"{i+20},{'M' if i%2==0 else 'F'},condition_{i%3},treatment_{i%2},outcome_{i%4}" 
                 for i in range(60)])  # 60 rows for minimum size check
        temp_file.write(test_data)
        temp_file.close()
        
        try:
            dataset = Dataset.objects.using('datasets_db').create(
                name='Compliant Dataset',
                description='Dataset that should pass compliance checks',
                file_path=temp_file.name,
                file_size=len(test_data),
                medical_domain='cardiology',  # Valid domain
                patient_count=60,
                data_type='tabular',
                uploaded_by_id=self.auditor_user.id
            )
            
            response = self.client.get(
                reverse('audit:medical_data_analysis'),
                {'dataset_id': dataset.id}
            )
            self.assertEqual(response.status_code, 200)
            
            # Should show compliance analysis
            self.assertContains(response, 'Medical Compliance Analysis')
            self.assertContains(response, 'Valid Medical Domain')
            self.assertContains(response, 'Minimum Dataset Size')
            
        finally:
            os.unlink(temp_file.name)


class SecurityAndWatermarkingTest(TestCase):
    """Test security features and watermarking."""
    
    databases = ['default', 'datasets_db']
    
    def setUp(self):
        """Set up test environment."""
        self.auditor_role = Role.objects.get(name='AUDITOR')
        self.auditor_user = CustomUser.objects.create_user(
            username='auditor_user',
            password='StrongPass123!',
            role=self.auditor_role
        )
        self.client = Client()
        self.client.login(username='auditor_user', password='StrongPass123!')

    def test_watermark_uniqueness(self):
        """Test that each access generates a unique watermark."""
        # Create minimal test dataset
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write("col1,col2\nval1,val2")
        temp_file.close()
        
        try:
            dataset = Dataset.objects.using('datasets_db').create(
                name='Test Dataset',
                description='Test dataset',
                file_path=temp_file.name,
                file_size=20,
                medical_domain='test',
                patient_count=1,
                data_type='tabular',
                uploaded_by_id=self.auditor_user.id
            )
            
            # First access
            response1 = self.client.get(
                reverse('audit:medical_data_analysis'),
                {'dataset_id': dataset.id}
            )
            self.assertEqual(response1.status_code, 200)
            
            # Second access
            response2 = self.client.get(
                reverse('audit:medical_data_analysis'),
                {'dataset_id': dataset.id}
            )
            self.assertEqual(response2.status_code, 200)
            
            # Both should have watermarks but different hashes
            self.assertContains(response1, 'HASH:')
            self.assertContains(response2, 'HASH:')
            
            # Extract hash from responses (this is a simplified check)
            content1 = response1.content.decode()
            content2 = response2.content.decode()
            
            # Both should contain the auditor username
            self.assertIn(self.auditor_user.username.upper(), content1)
            self.assertIn(self.auditor_user.username.upper(), content2)
            
        finally:
            os.unlink(temp_file.name)

    def test_security_warning_displayed(self):
        """Test that security warnings are prominently displayed."""
        response = self.client.get(reverse('audit:medical_data_analysis'))
        self.assertEqual(response.status_code, 200)
        
        # Check security warnings
        self.assertContains(response, 'RESTRICTED ACCESS - MEDICAL DATA')
        self.assertContains(response, 'This access is logged and monitored')
        self.assertContains(response, 'Maximum 100 rows per dataset')
        self.assertContains(response, 'All activity is recorded for compliance')

    def test_print_css_watermark_preservation(self):
        """Test that watermarks are preserved in print CSS."""
        response = self.client.get(reverse('audit:medical_data_analysis'))
        self.assertEqual(response.status_code, 200)
        
        content = response.content.decode()
        
        # Check print CSS rules for watermark preservation
        self.assertIn('media="print"', content)
        self.assertIn('color-adjust: exact', content)
        self.assertIn('display: block !important', content)

    def test_javascript_security_features(self):
        """Test JavaScript security features."""
        response = self.client.get(reverse('audit:medical_data_analysis'))
        self.assertEqual(response.status_code, 200)
        
        content = response.content.decode()
        
        # Check for security JavaScript
        self.assertIn('contextmenu', content)  # Right-click prevention
        self.assertIn('beforeunload', content)  # Page unload warning
        self.assertIn('beforeprint', content)  # Print watermarking
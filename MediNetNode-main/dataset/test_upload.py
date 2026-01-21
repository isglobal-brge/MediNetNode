#!/usr/bin/env python
"""
Test script for dataset upload functionality
"""
import os
import sys
import django
from pathlib import Path

# Add the project directory to the Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medinet.settings')
django.setup()

# Now import Django models
from django.contrib.auth import get_user_model
from dataset.uploader import SecureDatasetUploader
from dataset.models import Dataset

User = get_user_model()

def test_dataset_upload():
    """Test the dataset upload functionality"""
    print("Testing dataset upload functionality...")
    
    try:
        # Get or create a test user
        test_user, created = User.objects.get_or_create(
            username='testuser',
            defaults={
                'email': 'test@example.com',
                'is_active': True
            }
        )
        if created:
            test_user.set_password('testpass123')
            test_user.save()
        
        print(f"Using test user: {test_user.username}")
        
        # Path to the CSV file
        csv_path = "C:\\Users\\fraud\\Desktop\\Projects\\MediNetClient\\test_heart_failure.csv"
        
        if not os.path.exists(csv_path):
            print(f"[ERROR] CSV file not found at: {csv_path}")
            return False
            
        print(f"[OK] CSV file found: {csv_path}")
        
        # Create uploader
        uploader = SecureDatasetUploader(test_user)
        
        # Test upload with timestamp to ensure unique name
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"Heart Failure Clinical Records Test {timestamp}"
        
        print("Starting upload...")
        dataset = uploader.upload_dataset(
            file_path=csv_path,
            name=dataset_name,
            description="Test upload of heart failure clinical records dataset for federated learning",
            medical_domain="cardiology",
            data_type="tabular"
        )
        
        print(f"[SUCCESS] Upload successful! Dataset ID: {dataset.id}")
        print(f"   Name: {dataset.name}")
        print(f"   File size: {dataset.file_size} bytes")
        print(f"   Rows: {dataset.rows_count}")
        print(f"   Columns: {dataset.columns_count}")
        
        # Check if dataset is in database
        saved_dataset = Dataset.objects.using('datasets_db').get(id=dataset.id)
        print(f"[SUCCESS] Dataset verified in database: {saved_dataset.name}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Upload failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset_upload()
    sys.exit(0 if success else 1)
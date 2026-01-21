# MediNet RESEARCHER API Documentation

## Overview
The MediNet API provides stateless authentication endpoints for RESEARCHER users to access authorized medical datasets for federated learning. This API is compatible with the existing `client_api.py` structure and supports secure, audited access to medical data.

## Authentication
The API uses stateless authentication with API keys and IP whitelisting for maximum security.

### Required Headers
```http
X-API-Key: <your_api_key>
X-Client-IP: <your_whitelisted_ip>
Content-Type: application/json
```

## Base URL
```
http://localhost:8000/api/v1/
```

## Endpoints

### 1. Health Check
**GET** `/ping`

Simple health check endpoint to verify API connectivity.

#### Example Request
```bash
curl -X GET "http://localhost:8000/api/v1/ping" \
     -H "X-API-Key: rk_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6" \
     -H "X-Client-IP: 192.168.1.100"
```

#### Example Response
```json
{
    "status": "pong"
}
```

### 2. Get Dataset Information
**GET** `/get-data-info`

Retrieve metadata for all datasets accessible to the authenticated RESEARCHER user.

#### Example Request
```bash
curl -X GET "http://localhost:8000/api/v1/get-data-info" \
     -H "X-API-Key: rk_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6" \
     -H "X-Client-IP: 192.168.1.100"
```

#### Example Response
```json
{
    "dataset_id": [1, 2],
    "dataset_name": ["Heart Failure Clinical Records", "Diabetes Prediction Dataset"],
    "medical_domain": ["Cardiology", "Endocrinology"],
    "patient_count": [299, 768],
    "data_type": ["Tabular Data", "Tabular Data"],
    "file_size": [15420, 23876],
    "description": [
        "Clinical records of heart failure patients with survival analysis",
        "Diabetes prediction based on diagnostic measurements"
    ],
    "created_at": ["2024-01-15T10:30:00Z", "2024-01-20T14:45:00Z"]
}
```

#### Response Fields
- **dataset_id**: Array of unique dataset identifiers
- **dataset_name**: Array of human-readable dataset names
- **medical_domain**: Array of medical specialties (Cardiology, Neurology, etc.)
- **patient_count**: Array of patient counts per dataset
- **data_type**: Array of data types (Tabular Data, Image Data, etc.)
- **file_size**: Array of file sizes in bytes
- **description**: Array of dataset descriptions
- **created_at**: Array of creation timestamps in ISO format

### 3. Start Federated Learning Client
**POST** `/start-client`

Initiate federated learning training with the provided model configuration.

#### Example Request
```bash
curl -X POST "http://localhost:8000/api/v1/start-client" \
     -H "X-API-Key: rk_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6" \
     -H "X-Client-IP: 192.168.1.100" \
     -H "Content-Type: application/json" \
     -d '{
         "dataset_id": 1,
         "model_config": {
             "architecture": "neural_network",
             "layers": [
                 {"type": "dense", "units": 64, "activation": "relu"},
                 {"type": "dense", "units": 32, "activation": "relu"},
                 {"type": "dense", "units": 1, "activation": "sigmoid"}
             ],
             "optimizer": "adam",
             "loss": "binary_crossentropy",
             "metrics": ["accuracy"]
         },
         "training_params": {
             "epochs": 10,
             "batch_size": 32,
             "learning_rate": 0.001
         }
     }'
```

#### Example Response
```json
{
    "status": "success",
    "message": "Federated learning client started successfully",
    "client_id": "fl_client_abc123",
    "training_session_id": "session_def456"
}
```

## Example Python Script

Below is a complete Python example showing how to use the API to retrieve dataset information for the Heart Failure dataset:

```python
#!/usr/bin/env python3
"""
MediNet API Example Script
Demonstrates how to authenticate and retrieve dataset information.
"""

import requests
import json
from datetime import datetime

class MediNetAPIClient:
    def __init__(self, base_url, api_key, client_ip):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'X-API-Key': api_key,
            'X-Client-IP': client_ip,
            'Content-Type': 'application/json'
        }
    
    def ping(self):
        """Test API connectivity."""
        try:
            response = requests.get(
                f"{self.base_url}/ping",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def get_dataset_info(self):
        """Retrieve authorized dataset information."""
        try:
            response = requests.get(
                f"{self.base_url}/get-data-info",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def start_client(self, dataset_id, model_config, training_params):
        """Start federated learning client."""
        payload = {
            'dataset_id': dataset_id,
            'model_config': model_config,
            'training_params': training_params
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/start-client",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

def main():
    # Configuration
    BASE_URL = "http://localhost:8000/api/v1"
    API_KEY = "rk_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"  # Test API key
    CLIENT_IP = "192.168.1.100"  # Your whitelisted IP
    
    # Initialize client
    client = MediNetAPIClient(BASE_URL, API_KEY, CLIENT_IP)
    
    print("=== MediNet API Test Script ===\\n")
    
    # 1. Test connectivity
    print("1. Testing API connectivity...")
    ping_result = client.ping()
    if "error" in ping_result:
        print(f"❌ Ping failed: {ping_result['error']}")
        return
    print(f"✅ Ping successful: {ping_result['status']}")
    
    # 2. Get dataset information
    print("\\n2. Retrieving dataset information...")
    dataset_info = client.get_dataset_info()
    if "error" in dataset_info:
        print(f"❌ Failed to retrieve datasets: {dataset_info['error']}")
        return
    
    print("✅ Dataset information retrieved successfully:")
    print(f"   Found {len(dataset_info.get('dataset_id', []))} accessible datasets")
    
    # Display dataset details
    for i, dataset_id in enumerate(dataset_info.get('dataset_id', [])):
        print(f"\\n   Dataset {i+1}:")
        print(f"     ID: {dataset_id}")
        print(f"     Name: {dataset_info['dataset_name'][i]}")
        print(f"     Domain: {dataset_info['medical_domain'][i]}")
        print(f"     Patients: {dataset_info['patient_count'][i]}")
        print(f"     Type: {dataset_info['data_type'][i]}")
        print(f"     Size: {dataset_info['file_size'][i]:,} bytes")
        print(f"     Description: {dataset_info['description'][i]}")
        print(f"     Created: {dataset_info['created_at'][i]}")
    
    # 3. Start federated learning (example with Heart Failure dataset)
    if dataset_info.get('dataset_id'):
        heart_failure_id = dataset_info['dataset_id'][0]  # Use first dataset
        
        print(f"\\n3. Starting federated learning with dataset ID {heart_failure_id}...")
        
        model_config = {
            "architecture": "neural_network",
            "layers": [
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dropout", "rate": 0.2},
                {"type": "dense", "units": 32, "activation": "relu"},
                {"type": "dropout", "rate": 0.2},
                {"type": "dense", "units": 1, "activation": "sigmoid"}
            ],
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "metrics": ["accuracy", "precision", "recall"]
        }
        
        training_params = {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "validation_split": 0.2
        }
        
        start_result = client.start_client(heart_failure_id, model_config, training_params)
        
        if "error" in start_result:
            print(f"❌ Failed to start client: {start_result['error']}")
        else:
            print("✅ Federated learning client started successfully!")
            print(f"   Client ID: {start_result.get('client_id', 'N/A')}")
            print(f"   Session ID: {start_result.get('training_session_id', 'N/A')}")
    
    print("\\n=== Script completed ===")

if __name__ == "__main__":
    main()
```

## Expected Output

When running the script above, you should see output similar to:

```
=== MediNet API Test Script ===

1. Testing API connectivity...
✅ Ping successful: pong

2. Retrieving dataset information...
✅ Dataset information retrieved successfully:
   Found 1 accessible datasets

   Dataset 1:
     ID: 1
     Name: Heart Failure Clinical Records
     Domain: Cardiology
     Patients: 299
     Type: Tabular Data
     Size: 15,420 bytes
     Description: Clinical records of heart failure patients with survival analysis
     Created: 2024-01-15T10:30:00Z

3. Starting federated learning with dataset ID 1...
✅ Federated learning client started successfully!
   Client ID: fl_client_abc123
   Session ID: session_def456

=== Script completed ===
```

## Test Account Setup

To test the API, you need a RESEARCHER account with the following credentials:

### Test Account Details
- **Username**: `researcher_test`
- **Password**: `ResearcherPass123!`
- **Role**: RESEARCHER
- **API Key**: `rk_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`
- **Whitelisted IPs**: `192.168.1.100`, `10.0.0.50`

### Create Test Account (Admin Only)
```python
# Run this in Django shell: python manage.py shell
from users.models import CustomUser, Role, APIKey

# Create or get RESEARCHER role
role, _ = Role.objects.get_or_create(
    name='RESEARCHER',
    defaults={'permissions': {'api.access': True, 'dataset.view': True}}
)

# Create test user
user, created = CustomUser.objects.get_or_create(
    username='researcher_test',
    defaults={
        'email': 'researcher@test.com',
        'role': role
    }
)
if created:
    user.set_password('ResearcherPass123!')
    user.save()

# Create API key
api_key, created = APIKey.objects.get_or_create(
    user=user,
    name='Test API Key',
    defaults={
        'ip_whitelist': ['192.168.1.100', '10.0.0.50'],
        'rate_limit': 100
    }
)

print(f"API Key: {api_key.key}")
```

## Error Responses

### Authentication Errors
```json
{
    "error": "API authentication required",
    "status": 401
}
```

### Authorization Errors
```json
{
    "error": "No datasets available for this user",
    "status": 403
}
```

### Rate Limiting
```json
{
    "error": "Rate limit exceeded. Maximum 100 requests per hour.",
    "status": 429
}
```

### Server Errors
```json
{
    "error": "Internal server error retrieving dataset information",
    "status": 500
}
```

## Security Features

1. **Stateless Authentication**: No session management required
2. **IP Whitelisting**: API keys are restricted to specific IP addresses
3. **Rate Limiting**: 100 requests per hour per API key
4. **Comprehensive Auditing**: All API requests are logged with user, IP, and timestamp
5. **Role-Based Access**: Only RESEARCHER role can access API endpoints
6. **Dataset Authorization**: Users can only access datasets they have explicit permission for

## Dataset Structure Example

The Heart Failure Clinical Records dataset contains the following columns:

```
age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,
high_blood_pressure,platelets,serum_creatinine,serum_sodium,
sex,smoking,time,DEATH_EVENT
```

Sample data preview:
- **299 patients** with heart failure
- **13 clinical features** including age, medical conditions, lab values
- **Binary target** (DEATH_EVENT): 1 = death, 0 = survival
- **Time feature**: Follow-up period in days
- **Medical domain**: Cardiology
- **Use case**: Survival analysis and risk prediction

## Support

For API support, authentication issues, or dataset access requests:
- **Technical Issues**: Contact system administrator
- **Dataset Access**: Submit access request through admin interface
- **API Key Management**: Available in user profile after RESEARCHER role assignment

---

**⚠️ Security Notice**: This API provides access to sensitive medical data. All access is logged and monitored. Ensure compliance with healthcare data protection regulations (HIPAA, GDPR) when using this API.
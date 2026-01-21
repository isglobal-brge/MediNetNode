#!/usr/bin/env python
"""
Direct script to run permissions audit.
"""
import os
import sys
import django

# Add the project directory to the Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medinet.settings')

try:
    django.setup()
except Exception as e:
    print(f"Error setting up Django: {e}")
    sys.exit(1)

# Import after Django setup
from users.models import Role, CustomUser, APIKey

def audit_permissions():
    """Run a comprehensive permissions audit."""
    print("=== PERMISSION SYSTEM AUDIT ===")
    
    issues = []
    
    # 1. Check role permissions
    print("\n--- Role Permissions Audit ---")
    expected_permissions = {
        'RESEARCHER': ['api.access', 'dataset.view', 'dataset.train'],
        'ADMIN': ['system.admin', 'user.create', 'user.edit', 'dataset.create'],
        'AUDITOR': ['audit.view', 'training.view', 'user.view'],
    }
    
    for role_name, required_perms in expected_permissions.items():
        try:
            role = Role.objects.get(name=role_name)
            if not role.permissions:
                issues.append(f"Role {role_name} has empty permissions")
                print(f"[ERROR] {role_name} has empty permissions")
            else:
                missing_perms = []
                for perm in required_perms:
                    if not role.permissions.get(perm):
                        missing_perms.append(perm)
                
                if missing_perms:
                    issues.append(f"Role {role_name} missing permissions: {missing_perms}")
                    print(f"[WARNING] {role_name} missing: {missing_perms}")
                else:
                    print(f"[OK] {role_name} has required permissions")
        except Role.DoesNotExist:
            issues.append(f"Role {role_name} does not exist")
            print(f"[ERROR] Role {role_name} does not exist")
    
    # 2. Check user role assignments
    print("\n--- User Role Assignments ---")
    users_without_roles = CustomUser.objects.filter(is_active=True, role__isnull=True)
    if users_without_roles.exists():
        usernames = list(users_without_roles.values_list('username', flat=True))
        issues.append(f"Active users without roles: {usernames}")
        print(f"[WARNING] Users without roles: {usernames}")
    
    # User distribution
    active_users = CustomUser.objects.filter(is_active=True).exclude(role__isnull=True)
    role_counts = {}
    for user in active_users:
        role_name = user.role.name if user.role else 'None'
        role_counts[role_name] = role_counts.get(role_name, 0) + 1
    
    print("User role distribution:")
    for role, count in role_counts.items():
        print(f"  - {role}: {count} users")
    
    # 3. API key audit
    print("\n--- API Key Security Audit ---") 
    researcher_users = CustomUser.objects.filter(role__name='RESEARCHER', is_active=True)
    api_key_users = set(APIKey.objects.filter(is_active=True).values_list('user_id', flat=True))
    
    researchers_without_api = researcher_users.exclude(id__in=api_key_users)
    if researchers_without_api.exists():
        usernames = list(researchers_without_api.values_list('username', flat=True))
        print(f"[INFO] RESEARCHER users without API keys: {usernames}")
    
    non_researcher_api_keys = APIKey.objects.filter(is_active=True).exclude(user__role__name='RESEARCHER')
    if non_researcher_api_keys.exists():
        users = list(non_researcher_api_keys.values_list('user__username', flat=True))
        issues.append(f"Non-RESEARCHER users with API keys: {users}")
        print(f"[WARNING] Non-RESEARCHER users with API keys: {users}")
    
    # 4. Critical permissions check
    print("\n--- Critical Security Patterns ---")
    
    # RESEARCHER should not have web access
    researcher_users = CustomUser.objects.filter(role__name='RESEARCHER', is_active=True)
    for user in researcher_users:
        if user.has_permission('web.access'):
            issues.append(f"RESEARCHER {user.username} has web.access (should be API-only)")
            print(f"[ERROR] RESEARCHER {user.username} has web.access permission")
    
    # Show permission matrix
    print("\n--- Current Permission Matrix ---")
    roles = Role.objects.all().order_by('name')
    for role in roles:
        print(f"\n{role.name}:")
        if role.permissions:
            for perm, value in sorted(role.permissions.items()):
                status = "[OK]" if value else "[OFF]"
                print(f"  {status} {perm}")
        else:
            print("  (no permissions)")
    
    # Summary
    if issues:
        print(f"\n[AUDIT RESULT] Found {len(issues)} security issues:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        return False
    else:
        print("\n[AUDIT RESULT] No security issues found! [OK]")
        return True

if __name__ == '__main__':
    success = audit_permissions()
    sys.exit(0 if success else 1)
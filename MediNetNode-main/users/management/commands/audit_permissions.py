"""
Management command to audit and validate role permissions system.
Ensures all roles have proper permissions and detects security gaps.
"""
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from users.models import Role, APIKey

User = get_user_model()


class Command(BaseCommand):
    help = 'Audit role permissions system and detect security gaps'

    def add_arguments(self, parser):
        parser.add_argument(
            '--fix',
            action='store_true',
            help='Fix detected security issues automatically',
        )

    def handle(self, *args, **options):
        """Audit the permission system for security gaps."""
        auto_fix = options.get('fix', False)
        
        self.stdout.write("=== PERMISSION SYSTEM AUDIT ===")
        
        # 1. Check role permissions
        issues_found = []
        issues_found.extend(self._audit_role_permissions())
        
        # 2. Check user role assignments  
        issues_found.extend(self._audit_user_roles())
        
        # 3. Check API key permissions
        issues_found.extend(self._audit_api_permissions())
        
        # 4. Check critical permission patterns
        issues_found.extend(self._audit_critical_permissions())
        
        # Summary
        if issues_found:
            self.stdout.write(
                self.style.ERROR(f"\n[SECURITY AUDIT] Found {len(issues_found)} issues:")
            )
            for i, issue in enumerate(issues_found, 1):
                self.stdout.write(f"{i}. {issue}")
                
            if auto_fix:
                self.stdout.write("\n[AUTO-FIX] Attempting to fix issues...")
                self._fix_common_issues()
        else:
            self.stdout.write(
                self.style.SUCCESS("\n[SECURITY AUDIT] No permission issues found!")
            )
            
        # 5. Show current permission matrix
        self._show_permission_matrix()

    def _audit_role_permissions(self):
        """Check that all roles have proper permissions."""
        issues = []
        
        self.stdout.write("\n--- Role Permissions Audit ---")
        
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
                else:
                    missing_perms = []
                    for perm in required_perms:
                        if not role.permissions.get(perm):
                            missing_perms.append(perm)
                    
                    if missing_perms:
                        issues.append(f"Role {role_name} missing permissions: {missing_perms}")
                    else:
                        self.stdout.write(f"[OK] {role_name} has required permissions")
            except Role.DoesNotExist:
                issues.append(f"Role {role_name} does not exist")
        
        return issues

    def _audit_user_roles(self):
        """Check that all active users have roles assigned."""
        issues = []
        
        self.stdout.write("\n--- User Role Assignments Audit ---")
        
        users_without_roles = User.objects.filter(is_active=True, role__isnull=True)
        if users_without_roles.exists():
            usernames = list(users_without_roles.values_list('username', flat=True))
            issues.append(f"Active users without roles: {usernames}")
        
        # Check for users with inactive roles
        inactive_role_users = User.objects.filter(is_active=True).exclude(role__isnull=True)
        role_counts = {}
        for user in inactive_role_users:
            role_name = user.role.name if user.role else 'None'
            role_counts[role_name] = role_counts.get(role_name, 0) + 1
        
        self.stdout.write("User role distribution:")
        for role, count in role_counts.items():
            self.stdout.write(f"  - {role}: {count} users")
        
        return issues

    def _audit_api_permissions(self):
        """Check API key permissions and access patterns."""
        issues = []
        
        self.stdout.write("\n--- API Key Permissions Audit ---")
        
        # Check RESEARCHER users with API keys
        researcher_users = User.objects.filter(role__name='RESEARCHER', is_active=True)
        api_key_users = set(APIKey.objects.filter(is_active=True).values_list('user_id', flat=True))
        
        researchers_without_api = researcher_users.exclude(id__in=api_key_users)
        if researchers_without_api.exists():
            usernames = list(researchers_without_api.values_list('username', flat=True))
            self.stdout.write(f"RESEARCHER users without API keys: {usernames}")
        
        # Check for API keys with non-RESEARCHER users
        non_researcher_api_keys = APIKey.objects.filter(is_active=True).exclude(user__role__name='RESEARCHER')
        if non_researcher_api_keys.exists():
            users = list(non_researcher_api_keys.values_list('user__username', flat=True))
            issues.append(f"Non-RESEARCHER users with API keys: {users}")
        
        # Check API key security
        expired_keys = APIKey.objects.filter(is_active=True)
        expired_count = sum(1 for key in expired_keys if key.is_expired())
        if expired_count > 0:
            issues.append(f"{expired_count} expired API keys are still active")
        
        return issues

    def _audit_critical_permissions(self):
        """Check for critical permission patterns that could indicate security issues."""
        issues = []
        
        self.stdout.write("\n--- Critical Permissions Audit ---")
        
        # Check that RESEARCHER users can't access web interfaces
        researcher_users = User.objects.filter(role__name='RESEARCHER', is_active=True)
        for user in researcher_users:
            if user.has_permission('web.access'):
                issues.append(f"RESEARCHER {user.username} has web.access permission (should be API-only)")
        
        # Check that AUDITOR users can't modify data
        auditor_users = User.objects.filter(role__name='AUDITOR', is_active=True) 
        dangerous_perms = ['user.create', 'user.delete', 'dataset.create', 'dataset.delete']
        for user in auditor_users:
            for perm in dangerous_perms:
                if user.has_permission(perm):
                    issues.append(f"AUDITOR {user.username} has dangerous permission: {perm}")
        
        return issues

    def _fix_common_issues(self):
        """Fix common permission issues automatically."""
        try:
            from fix_researcher_permissions import fix_all_role_permissions
            success, updated_roles = fix_all_role_permissions()
            if success:
                self.stdout.write(
                    self.style.SUCCESS(f"[OK] Fixed role permissions for: {', '.join(updated_roles)}")
                )
            else:
                self.stdout.write(
                    self.style.ERROR("✗ Failed to fix role permissions")
                )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"✗ Error during auto-fix: {e}")
            )

    def _show_permission_matrix(self):
        """Display current permission matrix for all roles."""
        self.stdout.write("\n--- Current Permission Matrix ---")
        
        roles = Role.objects.all().order_by('name')
        if not roles.exists():
            self.stdout.write("No roles found!")
            return
        
        for role in roles:
            self.stdout.write(f"\n{role.name}:")
            if role.permissions:
                for perm, value in sorted(role.permissions.items()):
                    status = "[OK]" if value else "✗"
                    self.stdout.write(f"  {status} {perm}")
            else:
                self.stdout.write("  (no permissions)")
        
        # Show permission coverage
        all_perms = set()
        for role in roles:
            if role.permissions:
                all_perms.update(role.permissions.keys())
        
        self.stdout.write(f"\nTotal unique permissions in system: {len(all_perms)}")
        self.stdout.write("Permission categories:")
        
        categories = {}
        for perm in all_perms:
            category = perm.split('.')[0]
            categories[category] = categories.get(category, 0) + 1
        
        for category, count in sorted(categories.items()):
            self.stdout.write(f"  - {category}: {count} permissions")
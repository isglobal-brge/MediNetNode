from datetime import timedelta

from django.test import TestCase
from django.utils import timezone

from users.models import CustomUser, Role


class CustomUserModelTests(TestCase):
    def setUp(self) -> None:
        self.admin_role, _ = Role.objects.get_or_create(
            name='ADMIN', defaults={'permissions': {'user.create': True, 'user.view': True}}
        )
        self.researcher_role, _ = Role.objects.get_or_create(
            name='RESEARCHER', defaults={'permissions': {'user.view': True}}
        )

    def test_create_user_with_role(self) -> None:
        user = CustomUser.objects.create_user(
            username='test_admin', password='StrongPass123!', role=self.admin_role
        )
        self.assertEqual(user.role, self.admin_role)
        self.assertTrue(user.check_password('StrongPass123!'))

    def test_has_permission(self) -> None:
        user = CustomUser.objects.create_user(
            username='perm_user', password='StrongPass123!', role=self.admin_role
        )
        self.assertTrue(user.has_permission('user.create'))
        self.assertFalse(user.has_permission('non.existent'))

    def test_failed_attempts_increment_and_reset(self) -> None:
        user = CustomUser.objects.create_user(
            username='lock_user', password='StrongPass123!', role=self.researcher_role
        )
        for _ in range(3):
            user.increment_failed_attempts()
        user.refresh_from_db()
        self.assertEqual(user.failed_login_attempts, 3)

        user.reset_failed_attempts()
        user.refresh_from_db()
        self.assertEqual(user.failed_login_attempts, 0)
        self.assertIsNone(user.account_locked_until)

    def test_is_account_locked(self) -> None:
        user = CustomUser.objects.create_user(
            username='locked_user', password='StrongPass123!', role=self.researcher_role
        )
        user.account_locked_until = timezone.now() + timedelta(minutes=5)
        user.save(update_fields=['account_locked_until'])
        self.assertTrue(user.is_account_locked())

        user.account_locked_until = timezone.now() - timedelta(minutes=1)
        user.save(update_fields=['account_locked_until'])
        self.assertFalse(user.is_account_locked())

    def test_create_user_with_each_role(self):
        """Test: Crear usuario con cada rol (Admin, Investigador, Auditor)"""
        # Get or create roles (some might exist from setUp)
        admin_role, _ = Role.objects.get_or_create(
            name='ADMIN', 
            defaults={'permissions': {'user.create': True, 'user.view': True, 'user.delete': True}}
        )
        researcher_role, _ = Role.objects.get_or_create(
            name='RESEARCHER', 
            defaults={'permissions': {'user.view': True}}
        )
        auditor_role, _ = Role.objects.get_or_create(
            name='AUDITOR', 
            defaults={'permissions': {'logs.view': True, 'audit.view': True}}
        )
        
        # Test Admin user creation
        admin_user = CustomUser.objects.create_user(
            username='admin_user',
            password='AdminPass123!',
            email='admin@example.com',
            role=admin_role
        )
        self.assertEqual(admin_user.role.name, 'ADMIN')
        self.assertTrue(admin_user.has_permission('user.create'))
        self.assertTrue(admin_user.has_permission('user.view'))
        
        # Test Investigador user creation
        researcher_user = CustomUser.objects.create_user(
            username='researcher_user',
            password='ResearcherPass123!',
            email='researcher@example.com',
            role=researcher_role
        )
        self.assertEqual(researcher_user.role.name, 'RESEARCHER')
        # RESEARCHER should NOT have user management permissions
        self.assertFalse(researcher_user.has_permission('user.view'))
        self.assertFalse(researcher_user.has_permission('user.create'))
        
        # Test Auditor user creation
        auditor_user = CustomUser.objects.create_user(
            username='auditor_user',
            password='AuditorPass123!',
            email='auditor@example.com',
            role=auditor_role
        )
        self.assertEqual(auditor_user.role.name, 'AUDITOR')
        self.assertTrue(auditor_user.has_permission('audit.view'))
        self.assertFalse(auditor_user.has_permission('user.create'))

    def test_session_expired(self):
        """Test session expiration functionality."""
        user = CustomUser.objects.create_user(
            username='session_user',
            password='SessionPass123!',
            role=self.admin_role
        )
        
        # New user without last_activity should be expired
        self.assertTrue(user.is_session_expired())
        
        # Set recent activity
        user.last_activity = timezone.now()
        user.save()
        self.assertFalse(user.is_session_expired())
        
        # Set old activity (more than 30 minutes ago)
        user.last_activity = timezone.now() - timedelta(minutes=35)
        user.save()
        self.assertTrue(user.is_session_expired())

    def test_password_history(self):
        """Test password history functionality."""
        from users.models import PasswordHistory
        
        user = CustomUser.objects.create_user(
            username='history_user',
            password='InitialPass123!',
            role=self.admin_role
        )
        initial_password_hash = user.password
        
        # Change password - should create history entry
        user.set_password('NewPass123!')
        user.save()
        
        # Check history was created
        history = PasswordHistory.objects.filter(user=user)
        self.assertEqual(history.count(), 1)
        self.assertEqual(history.first().password_hash, initial_password_hash)
        
        # Test password reuse detection
        self.assertTrue(user.check_password_history('InitialPass123!'))  # Old password
        self.assertTrue(user.check_password_history('NewPass123!'))      # Current password
        self.assertFalse(user.check_password_history('NeverUsedPass123!'))  # Never used
        
        # Change password 5 more times
        for i in range(5):
            user.set_password(f'Password{i}123!')
            user.save()
        
        # Should only keep last 5 passwords in history + current
        total_history = PasswordHistory.objects.filter(user=user).count()
        self.assertLessEqual(total_history, 5)  # Max 5 in history
        
        # Initial password should no longer be detectable after 6 changes
        self.assertFalse(user.check_password_history('InitialPass123!'))



"""
Tests for users views (TAREA 1.4 validation tests).
"""
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.contrib.messages import get_messages
from datetime import timedelta
from django.utils import timezone

from users.models import Role, PasswordHistory
from audit.models import AuditLog

User = get_user_model()


class UserViewsTests(TestCase):
    """Test user management views for TAREA 1.4."""
    
    # Add datasets_db to allowed databases for admin dashboard tests
    databases = {'default', 'datasets_db'}
    
    def setUp(self):
        # Create roles
        self.admin_role = Role.objects.get(name='ADMIN')
        self.investigator_role = Role.objects.get(name='RESEARCHER')
        self.auditor_role = Role.objects.get(name='AUDITOR')
        
        # Create admin user
        self.admin_user = User.objects.create_user(
            username='admin_test',
            password='AdminPass123!',
            email='admin@test.com',
            role=self.admin_role
        )
        
        # Create investigator user  
        self.investigator_user = User.objects.create_user(
            username='investigator_test',
            password='InvestigatorPass123!',
            email='investigator@test.com',
            role=self.investigator_role
        )
        
        self.client = Client()

    def test_admin_dashboard_accessible_only_for_admin(self):
        """Test: Admin dashboard access control"""
        # Not logged in -> redirect to login
        response = self.client.get('/admin/')
        self.assertEqual(response.status_code, 302)
        
        # Researcher -> redirected to info page
        self.client.login(username='investigator_test', password='InvestigatorPass123!')
        response = self.client.get('/admin/')
        self.assertEqual(response.status_code, 302)
        self.assertIn('/info/researcher/', response['Location'])
        
        # Admin -> accessible
        self.client.login(username='admin_test', password='AdminPass123!')
        response = self.client.get('/admin/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Dashboard')

    def test_researcher_info_view(self):
        """Researcher sees informative page; admins are redirected back to dashboard."""
        # Researcher
        self.client.login(username='investigator_test', password='InvestigatorPass123!')
        response = self.client.get('/info/researcher/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Limited Access')

        # Admin redirected
        self.client.login(username='admin_test', password='AdminPass123!')
        response = self.client.get('/info/researcher/')
        self.assertEqual(response.status_code, 302)
        self.assertIn('/admin/', response['Location'])

    def test_django_superuser_is_treated_as_admin(self):
        """A Django superuser (without role) can access admin dashboard."""
        User = get_user_model()
        su = User.objects.create_superuser(
            username='root_su', email='su@test.com', password='RootPass123!'
        )
        # Ensure no role assigned
        self.assertIsNone(su.role)
        self.client.login(username='root_su', password='RootPass123!')
        response = self.client.get('/admin/')
        self.assertEqual(response.status_code, 200)

    def test_create_user_generates_audit_log(self):
        """Test: Crear usuario genera log de auditoría"""
        self.client.login(username='admin_test', password='AdminPass123!')
        
        initial_log_count = AuditLog.objects.count()
        
        response = self.client.post('/users/create/', {
            'username': 'new_test_user',
            'email': 'newuser@test.com',
            'first_name': 'New',
            'last_name': 'User',
            'role': self.investigator_role.id,
            'password1': 'NewUserPass123!',
            'password2': 'NewUserPass123!',
        })
        
        # Should redirect after successful creation
        self.assertEqual(response.status_code, 302)
        
        # Check user was created
        self.assertTrue(User.objects.filter(username='new_test_user').exists())
        
        # Check audit log was created
        self.assertEqual(AuditLog.objects.count(), initial_log_count + 1)
        
        log = AuditLog.objects.latest('timestamp')
        self.assertEqual(log.action, 'USER_CREATE')
        self.assertEqual(log.user, self.admin_user)
        self.assertTrue(log.success)

    def test_form_validations_work_correctly(self):
        """Test: Validaciones de formulario funcionan correctamente"""
        self.client.login(username='admin_test', password='AdminPass123!')
        
        # Test with weak password
        response = self.client.post(reverse('create_user'), {
            'username': 'test_weak_pass',
            'email': 'weak@test.com',
            'first_name': 'Weak',
            'last_name': 'Pass',
            'role': self.investigator_role.id,
            'password1': 'weak',  # Too weak
            'password2': 'weak',
        })
        
        # Should not redirect (form has errors)
        self.assertEqual(response.status_code, 200)
        
        # User should not be created
        self.assertFalse(User.objects.filter(username='test_weak_pass').exists())
        
        # Test with duplicate email
        response = self.client.post(reverse('create_user'), {
            'username': 'test_duplicate',
            'email': 'admin@test.com',  # Duplicate email
            'first_name': 'Duplicate',
            'last_name': 'Email', 
            'role': self.investigator_role.id,
            'password1': 'ValidPass123!',
            'password2': 'ValidPass123!',
        })
        
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username='test_duplicate').exists())

    def test_search_and_filters_return_correct_results(self):
        """Test: Búsqueda y filtros devuelven resultados correctos"""
        self.client.login(username='admin_test', password='AdminPass123!')
        
        # Test search by username
        response = self.client.get(reverse('user_list'), {'q': 'admin'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'admin_test')
        self.assertNotContains(response, 'investigator_test')
        
        # Test filter by role
        response = self.client.get(reverse('user_list'), {'role': self.investigator_role.id})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'investigator_test')
        self.assertNotContains(response, 'admin_test')
        
        # Test filter by active status
        self.investigator_user.is_active = False
        self.investigator_user.save()
        
        response = self.client.get(reverse('user_list'), {'is_active': 'True'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'admin_test')
        self.assertNotContains(response, 'investigator_test')

    def test_role_change_requires_confirmation_and_generates_log(self):
        """Test: Cambio de rol requiere confirmación y genera log"""
        self.client.login(username='admin_test', password='AdminPass123!')
        
        initial_log_count = AuditLog.objects.count()
        
        # Update user role  
        response = self.client.post(reverse('update_user', args=[self.investigator_user.id]), {
            'username': self.investigator_user.username,
            'email': self.investigator_user.email,
            'first_name': 'Investigator',  # Provide required name
            'last_name': 'Test',  # Provide required surname
            'role': self.auditor_role.id,  # Change role
            'is_active': True,
        })
        
        # Should redirect after successful update
        self.assertEqual(response.status_code, 302)
        
        # Check role was changed
        self.investigator_user.refresh_from_db()
        self.assertEqual(self.investigator_user.role, self.auditor_role)
        
        # Check audit log was created
        self.assertEqual(AuditLog.objects.count(), initial_log_count + 1)
        
        log = AuditLog.objects.latest('timestamp')
        self.assertEqual(log.action, 'USER_UPDATE')
        self.assertEqual(log.user, self.admin_user)
        self.assertTrue(log.success)

    def test_user_cannot_change_own_role(self):
        """Test: Usuario no puede cambiar su propio rol"""
        self.client.login(username='admin_test', password='AdminPass123!')
        
        # Try to change own role
        response = self.client.post(reverse('update_user', args=[self.admin_user.id]), {
            'username': self.admin_user.username,
            'email': self.admin_user.email,
            'first_name': self.admin_user.first_name,
            'last_name': self.admin_user.last_name,
            'role': self.investigator_role.id,  # Try to change own role
            'is_active': True,
        })
        
        # Should show form with error
        self.assertEqual(response.status_code, 200)
        
        # Role should not change
        self.admin_user.refresh_from_db() 
        self.assertEqual(self.admin_user.role, self.admin_role)

    def test_user_export_csv_functionality(self):
        """Test: Funcionalidad de exportar usuarios a CSV"""
        self.client.login(username='admin_test', password='AdminPass123!')
        
        response = self.client.get(reverse('export_users_csv'))
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/csv')
        self.assertIn('attachment; filename="users.csv"', response['Content-Disposition'])
        
        # Check CSV content contains users
        content = response.content.decode('utf-8')
        self.assertIn('admin_test', content)
        self.assertIn('investigator_test', content)

    def test_permission_based_access_control(self):
        """Test: Control de acceso basado en permisos"""
        # Test investigator (RESEARCHER role) gets redirected from web pages
        self.client.login(username='investigator_test', password='InvestigatorPass123!')
        
        response = self.client.get(reverse('create_user'))
        self.assertEqual(response.status_code, 302)  # Redirected to researcher info
        
        # Test investigator also gets redirected from user list (RESEARCHER web blocking)
        response = self.client.get(reverse('user_list'))
        self.assertEqual(response.status_code, 302)  # Also redirected - no web access for RESEARCHER

    def test_user_detail_shows_security_information(self):
        """Test: Vista de detalle muestra información de seguridad"""
        self.client.login(username='admin_test', password='AdminPass123!')
        
        response = self.client.get(reverse('user_detail', args=[self.investigator_user.id]))
        self.assertEqual(response.status_code, 200)
        
        # Should show user information
        self.assertContains(response, self.investigator_user.username)
        self.assertContains(response, self.investigator_user.email)
        
        # Should show security status
        self.assertContains(response, 'RESEARCHER')  # Role badge

    def test_pagination_works_correctly(self):
        """Test: Paginación funciona correctamente"""
        # Create many users to test pagination
        for i in range(30):
            User.objects.create_user(
                username=f'test_user_{i}',
                password='TestPass123!',
                email=f'test{i}@test.com',
                role=self.investigator_role
            )
        
        self.client.login(username='admin_test', password='AdminPass123!')
        
        response = self.client.get(reverse('user_list'))
        self.assertEqual(response.status_code, 200)
        
        # Should have pagination
        self.assertContains(response, 'pagination')
        
        # Test second page
        response = self.client.get(reverse('user_list'), {'page': 2})
        self.assertEqual(response.status_code, 200)
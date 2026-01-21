"""
Tests to validate that RESEARCHER role is completely blocked from web access.
Critical security tests - RESEARCHER should only access via API.
"""
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from users.models import Role

User = get_user_model()


class ResearcherWebBlockingTests(TestCase):
    """Test that RESEARCHER role is completely blocked from web interface."""

    # Add datasets_db to allowed databases for admin dashboard tests
    databases = {'default', 'datasets_db'}

    def setUp(self):
        """Set up test users and roles."""
        # Get roles (already created by conftest.py session fixture)
        self.admin_role = Role.objects.get(name='ADMIN')
        self.researcher_role = Role.objects.get(name='RESEARCHER')

        # Create users
        self.admin_user = User.objects.create_user(
            username='admin_test',
            password='AdminPass123!',
            email='admin@test.com',
            role=self.admin_role
        )

        self.researcher_user = User.objects.create_user(
            username='researcher_test',
            password='ResearcherPass123!',
            email='researcher@test.com',
            role=self.researcher_role
        )

        self.client = Client()
    
    def test_researcher_blocked_from_admin_dashboard(self):
        """Test that RESEARCHER cannot access admin dashboard."""
        self.client.login(username='researcher_test', password='ResearcherPass123!')
        response = self.client.get(reverse('admin_dashboard'))
        
        # Should redirect to researcher info page
        self.assertEqual(response.status_code, 302)
        self.assertIn('/info/researcher/', response.url)
    
    def test_researcher_blocked_from_user_list(self):
        """Test that RESEARCHER cannot access user list."""
        self.client.login(username='researcher_test', password='ResearcherPass123!')
        response = self.client.get(reverse('user_list'))
        
        # Should redirect to researcher info page
        self.assertEqual(response.status_code, 302)
        self.assertIn('/info/researcher/', response.url)
    
    def test_researcher_blocked_from_dataset_views(self):
        """Test that RESEARCHER cannot access any dataset web views."""
        self.client.login(username='researcher_test', password='ResearcherPass123!')
        
        # Test dataset list
        response = self.client.get(reverse('dataset:list'))
        self.assertEqual(response.status_code, 302)
        self.assertIn('/info/researcher/', response.url)
        
        # Test dataset dashboard  
        response = self.client.get(reverse('dataset:dashboard'))
        self.assertEqual(response.status_code, 302)
        self.assertIn('/info/researcher/', response.url)
        
        # Test dataset upload
        response = self.client.get(reverse('dataset:upload'))
        self.assertEqual(response.status_code, 302)
        self.assertIn('/info/researcher/', response.url)
    
    def test_researcher_can_access_researcher_info_page(self):
        """Test that RESEARCHER can access their info page."""
        self.client.login(username='researcher_test', password='ResearcherPass123!')
        response = self.client.get(reverse('researcher_info'))
        
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Web Access Restricted')
        self.assertContains(response, 'API endpoints')
    
    def test_researcher_can_logout(self):
        """Test that RESEARCHER can logout."""
        self.client.login(username='researcher_test', password='ResearcherPass123!')
        response = self.client.get(reverse('logout'))
        
        # Should redirect to login page (normal logout behavior), not researcher info
        self.assertEqual(response.status_code, 302)
        self.assertNotIn('/info/researcher/', response.url)
    
    def test_researcher_blocked_from_user_creation(self):
        """Test that RESEARCHER cannot access user creation."""
        self.client.login(username='researcher_test', password='ResearcherPass123!')
        response = self.client.get(reverse('create_user'))
        
        self.assertEqual(response.status_code, 302)
        self.assertIn('/info/researcher/', response.url)
    
    def test_researcher_blocked_from_user_detail(self):
        """Test that RESEARCHER cannot access user details."""
        self.client.login(username='researcher_test', password='ResearcherPass123!')
        response = self.client.get(reverse('user_detail', kwargs={'user_id': self.admin_user.id}))
        
        self.assertEqual(response.status_code, 302)
        self.assertIn('/info/researcher/', response.url)
    
    def test_admin_can_access_all_views(self):
        """Control test: verify admin can still access everything."""
        self.client.login(username='admin_test', password='AdminPass123!')
        
        # Admin should access admin dashboard
        response = self.client.get(reverse('admin_dashboard'))
        self.assertEqual(response.status_code, 200)
        
        # Admin should access user list
        response = self.client.get(reverse('user_list'))
        self.assertEqual(response.status_code, 200)
        
        # Admin should access dataset dashboard
        response = self.client.get(reverse('dataset:dashboard'))
        self.assertEqual(response.status_code, 200)
    
    def test_researcher_info_page_content(self):
        """Test that researcher info page has correct content."""
        self.client.login(username='researcher_test', password='ResearcherPass123!')
        response = self.client.get(reverse('researcher_info'))
        
        self.assertEqual(response.status_code, 200)
        
        # Check for key security messages
        self.assertContains(response, 'RESEARCHER')
        self.assertContains(response, 'API endpoints')
        self.assertContains(response, 'IP-based authentication')
        self.assertContains(response, 'Web interface access is restricted')
    
    def test_researcher_cannot_post_to_any_form(self):
        """Test that RESEARCHER cannot POST to any web forms."""
        self.client.login(username='researcher_test', password='ResearcherPass123!')
        
        # Try to POST to user creation form
        response = self.client.post(reverse('create_user'), {
            'username': 'hacker_attempt',
            'email': 'hack@test.com',
            'password1': 'HackPass123!',
            'password2': 'HackPass123!'
        })
        
        # Should be redirected, not processed
        self.assertEqual(response.status_code, 302)
        self.assertIn('/info/researcher/', response.url)
        
        # Verify user was not created
        self.assertFalse(User.objects.filter(username='hacker_attempt').exists())
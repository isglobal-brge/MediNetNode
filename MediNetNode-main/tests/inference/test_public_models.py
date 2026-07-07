"""
Tests for public_models view.
"""
import pytest
from django.urls import reverse
from users.models import Role


@pytest.fixture
def member_role(db):
    """Create MEMBER role with inference permissions."""
    role, _ = Role.objects.get_or_create(
        name='MEMBER',
        defaults={
            'permissions': {
                'inference.execute': 'ALL',
                'inference.upload': True,
            }
        }
    )
    return role


@pytest.fixture
def admin_role(db):
    """Create ADMIN role."""
    role, _ = Role.objects.get_or_create(
        name='ADMIN',
        defaults={
            'permissions': {
                'inference.execute': 'ALL',
                'inference.upload': True,
                'inference.approve': True,
            }
        }
    )
    return role


@pytest.fixture
def member_user(db, django_user_model, member_role):
    """Create a MEMBER user."""
    user = django_user_model.objects.create_user(
        username='test_member_public',
        password='testpass123',
        email='member_public@test.com'
    )
    user.role = member_role
    user.save()
    return user


@pytest.fixture
def admin_user(db, django_user_model, admin_role):
    """Create an ADMIN user."""
    user = django_user_model.objects.create_user(
        username='test_admin_public',
        password='testpass123',
        email='admin_public@test.com'
    )
    user.role = admin_role
    user.save()
    return user


@pytest.mark.django_db
class TestPublicModelsView:
    """Tests for the public_models view."""

    def test_public_models_view_requires_login(self, client):
        """Test that public_models view requires authentication."""
        url = reverse('inference:public_models')
        response = client.get(url)
        assert response.status_code == 302
        assert '/auth/login/' in response.url

    def test_public_models_view_accessible_by_member(self, client, member_user):
        """Test that MEMBER can access public_models view."""
        client.force_login(member_user)
        url = reverse('inference:public_models')
        response = client.get(url)
        assert response.status_code == 200
        assert 'inference/public_models.html' in [t.name for t in response.templates]

    def test_public_models_view_accessible_by_admin(self, client, admin_user):
        """Test that ADMIN can access public_models view."""
        client.force_login(admin_user)
        url = reverse('inference:public_models')
        response = client.get(url)
        assert response.status_code == 200
        assert 'inference/public_models.html' in [t.name for t in response.templates]

    def test_public_models_view_has_correct_context(self, client, member_user):
        """Test that public_models view provides correct context."""
        client.force_login(member_user)
        url = reverse('inference:public_models')
        response = client.get(url)

        assert response.status_code == 200
        assert 'page_title' in response.context
        assert response.context['page_title'] == 'Public Models'
        assert 'models' in response.context
        assert 'total_count' in response.context
        assert 'total_public' in response.context
        assert 'domain_filter' in response.context
        assert 'search_query' in response.context
        assert 'sort_by' in response.context
        assert 'sort_choices' in response.context

    def test_public_models_view_filter_by_domain(self, client, member_user):
        """Test filtering by domain."""
        client.force_login(member_user)
        url = reverse('inference:public_models')
        response = client.get(url, {'domain': 'cardiology'})

        assert response.status_code == 200
        assert response.context['domain_filter'] == 'cardiology'

    def test_public_models_view_search(self, client, member_user):
        """Test search functionality."""
        client.force_login(member_user)
        url = reverse('inference:public_models')
        response = client.get(url, {'q': 'test search'})

        assert response.status_code == 200
        assert response.context['search_query'] == 'test search'

    def test_public_models_view_sort_options(self, client, member_user):
        """Test sorting functionality."""
        client.force_login(member_user)
        url = reverse('inference:public_models')

        # Test each sort option
        for sort_option in ['recent', 'popular', 'name', 'domain']:
            response = client.get(url, {'sort': sort_option})
            assert response.status_code == 200
            assert response.context['sort_by'] == sort_option

    def test_public_models_view_pagination(self, client, member_user):
        """Test pagination parameter is accepted."""
        client.force_login(member_user)
        url = reverse('inference:public_models')
        response = client.get(url, {'page': 1})

        assert response.status_code == 200

    def test_public_models_view_breadcrumbs(self, client, member_user):
        """Test that breadcrumbs are present."""
        client.force_login(member_user)
        url = reverse('inference:public_models')
        response = client.get(url)

        assert response.status_code == 200
        assert 'breadcrumbs' in response.context
        breadcrumbs = response.context['breadcrumbs']
        assert len(breadcrumbs) == 2
        assert breadcrumbs[0]['name'] == 'Dashboard'
        assert breadcrumbs[1]['name'] == 'Public Models'

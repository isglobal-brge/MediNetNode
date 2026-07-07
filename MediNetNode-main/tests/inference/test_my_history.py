"""
Tests for my_history view.
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
        username='test_member_history',
        password='testpass123',
        email='member_history@test.com'
    )
    user.role = member_role
    user.save()
    return user


@pytest.fixture
def admin_user(db, django_user_model, admin_role):
    """Create an ADMIN user."""
    user = django_user_model.objects.create_user(
        username='test_admin_history',
        password='testpass123',
        email='admin_history@test.com'
    )
    user.role = admin_role
    user.save()
    return user


@pytest.mark.django_db
class TestMyHistoryView:
    """Tests for the my_history view."""

    def test_my_history_view_requires_login(self, client):
        """Test that my_history view requires authentication."""
        url = reverse('inference:my_history')
        response = client.get(url)
        assert response.status_code == 302
        assert '/auth/login/' in response.url

    def test_my_history_view_accessible_by_member(self, client, member_user):
        """Test that MEMBER can access my_history view."""
        client.force_login(member_user)
        url = reverse('inference:my_history')
        response = client.get(url)
        assert response.status_code == 200
        assert 'inference/my_history.html' in [t.name for t in response.templates]

    def test_my_history_view_accessible_by_admin(self, client, admin_user):
        """Test that ADMIN can access my_history view."""
        client.force_login(admin_user)
        url = reverse('inference:my_history')
        response = client.get(url)
        assert response.status_code == 200
        assert 'inference/my_history.html' in [t.name for t in response.templates]

    def test_my_history_view_has_correct_context(self, client, member_user):
        """Test that my_history view provides correct context."""
        client.force_login(member_user)
        url = reverse('inference:my_history')
        response = client.get(url)

        assert response.status_code == 200
        assert 'page_title' in response.context
        assert response.context['page_title'] == 'My Prediction History'
        assert 'predictions' in response.context
        assert 'total_count' in response.context
        assert 'stats' in response.context
        assert 'model_filter' in response.context
        assert 'domain_filter' in response.context
        assert 'status_filter' in response.context
        assert 'search_query' in response.context
        assert 'sort_by' in response.context

    def test_my_history_view_stats_structure(self, client, member_user):
        """Test that stats have correct structure."""
        client.force_login(member_user)
        url = reverse('inference:my_history')
        response = client.get(url)

        stats = response.context['stats']
        assert 'total' in stats
        assert 'successful' in stats
        assert 'failed' in stats
        assert 'total_records' in stats
        assert 'avg_execution_time' in stats
        assert 'success_rate' in stats

    def test_my_history_view_filter_by_status(self, client, member_user):
        """Test filtering by status."""
        client.force_login(member_user)
        url = reverse('inference:my_history')

        # Test success filter
        response = client.get(url, {'status': 'success'})
        assert response.status_code == 200
        assert response.context['status_filter'] == 'success'

        # Test failed filter
        response = client.get(url, {'status': 'failed'})
        assert response.status_code == 200
        assert response.context['status_filter'] == 'failed'

    def test_my_history_view_filter_by_domain(self, client, member_user):
        """Test filtering by domain."""
        client.force_login(member_user)
        url = reverse('inference:my_history')
        response = client.get(url, {'domain': 'cardiology'})

        assert response.status_code == 200
        assert response.context['domain_filter'] == 'cardiology'

    def test_my_history_view_search(self, client, member_user):
        """Test search functionality."""
        client.force_login(member_user)
        url = reverse('inference:my_history')
        response = client.get(url, {'q': 'test model'})

        assert response.status_code == 200
        assert response.context['search_query'] == 'test model'

    def test_my_history_view_sort_options(self, client, member_user):
        """Test sorting functionality."""
        client.force_login(member_user)
        url = reverse('inference:my_history')

        # Test each sort option
        for sort_option in ['recent', 'oldest', 'model', 'records', 'time']:
            response = client.get(url, {'sort': sort_option})
            assert response.status_code == 200
            assert response.context['sort_by'] == sort_option

    def test_my_history_view_date_filters(self, client, member_user):
        """Test date filtering."""
        client.force_login(member_user)
        url = reverse('inference:my_history')

        response = client.get(url, {
            'date_from': '2024-01-01',
            'date_to': '2024-12-31'
        })

        assert response.status_code == 200
        assert response.context['date_from'] == '2024-01-01'
        assert response.context['date_to'] == '2024-12-31'

    def test_my_history_view_pagination(self, client, member_user):
        """Test pagination parameter is accepted."""
        client.force_login(member_user)
        url = reverse('inference:my_history')
        response = client.get(url, {'page': 1})

        assert response.status_code == 200

    def test_my_history_view_breadcrumbs(self, client, member_user):
        """Test that breadcrumbs are present."""
        client.force_login(member_user)
        url = reverse('inference:my_history')
        response = client.get(url)

        assert response.status_code == 200
        assert 'breadcrumbs' in response.context
        breadcrumbs = response.context['breadcrumbs']
        assert len(breadcrumbs) == 2
        assert breadcrumbs[0]['name'] == 'Dashboard'
        assert breadcrumbs[1]['name'] == 'My History'

    def test_my_history_view_invalid_date_handled(self, client, member_user):
        """Test that invalid dates are handled gracefully."""
        client.force_login(member_user)
        url = reverse('inference:my_history')

        # Invalid date format should not crash
        response = client.get(url, {'date_from': 'invalid-date'})
        assert response.status_code == 200

    def test_my_history_view_combined_filters(self, client, member_user):
        """Test multiple filters at once."""
        client.force_login(member_user)
        url = reverse('inference:my_history')

        response = client.get(url, {
            'status': 'success',
            'domain': 'cardiology',
            'sort': 'records',
            'q': 'test'
        })

        assert response.status_code == 200
        assert response.context['status_filter'] == 'success'
        assert response.context['domain_filter'] == 'cardiology'
        assert response.context['sort_by'] == 'records'
        assert response.context['search_query'] == 'test'

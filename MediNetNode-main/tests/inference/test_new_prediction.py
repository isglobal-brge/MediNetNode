"""
Tests for new_prediction and prediction_load_data views.
"""
import pytest
from django.urls import reverse
from users.models import Role
from inference.models import DeployedModel


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
        username='test_member_predict',
        password='testpass123',
        email='member_predict@test.com'
    )
    user.role = member_role
    user.save()
    return user


@pytest.fixture
def admin_user(db, django_user_model, admin_role):
    """Create an ADMIN user."""
    user = django_user_model.objects.create_user(
        username='test_admin_predict',
        password='testpass123',
        email='admin_predict@test.com'
    )
    user.role = admin_role
    user.save()
    return user


@pytest.fixture
def approved_model(db, member_user):
    """Create an approved model for testing."""
    return DeployedModel.objects.create(
        name='Test Prediction Model',
        version='1.0.0',
        description='A test model for predictions',
        domain='cardiology',
        uploaded_by=member_user,
        status='approved',
        is_public=False,
        input_schema={'feature_names': ['age', 'bp'], 'dtypes': {'age': 'int', 'bp': 'float'}},
        output_schema={'type': 'classification', 'classes': ['low', 'high']},
    )


@pytest.fixture
def public_model(db, admin_user):
    """Create a public approved model."""
    return DeployedModel.objects.create(
        name='Public Test Model',
        version='1.0.0',
        description='A public test model',
        domain='neurology',
        uploaded_by=admin_user,
        status='approved',
        is_public=True,
        input_schema={'feature_names': ['x', 'y'], 'dtypes': {}},
        output_schema={'type': 'regression'},
    )


@pytest.mark.django_db
class TestNewPredictionView:
    """Tests for the new_prediction view (Step 1)."""

    def test_new_prediction_view_requires_login(self, client):
        """Test that new_prediction view requires authentication."""
        url = reverse('inference:new_prediction')
        response = client.get(url)
        assert response.status_code == 302
        assert '/auth/login/' in response.url

    def test_new_prediction_view_accessible_by_member(self, client, member_user):
        """Test that MEMBER can access new_prediction view."""
        client.force_login(member_user)
        url = reverse('inference:new_prediction')
        response = client.get(url)
        assert response.status_code == 200
        assert 'inference/new_prediction.html' in [t.name for t in response.templates]

    def test_new_prediction_view_accessible_by_admin(self, client, admin_user):
        """Test that ADMIN can access new_prediction view."""
        client.force_login(admin_user)
        url = reverse('inference:new_prediction')
        response = client.get(url)
        assert response.status_code == 200

    def test_new_prediction_view_has_wizard_context(self, client, member_user):
        """Test that view provides wizard step context."""
        client.force_login(member_user)
        url = reverse('inference:new_prediction')
        response = client.get(url)

        assert response.status_code == 200
        assert 'wizard_step' in response.context
        assert response.context['wizard_step'] == 1
        assert 'wizard_steps' in response.context
        assert len(response.context['wizard_steps']) == 3

    def test_new_prediction_view_shows_user_models(self, client, member_user, approved_model):
        """Test that user's approved models are shown."""
        client.force_login(member_user)
        url = reverse('inference:new_prediction')
        response = client.get(url)

        assert response.status_code == 200
        assert 'my_models' in response.context
        assert approved_model in response.context['my_models']

    def test_new_prediction_view_shows_public_models(self, client, member_user, public_model):
        """Test that public approved models are shown."""
        client.force_login(member_user)
        url = reverse('inference:new_prediction')
        response = client.get(url)

        assert response.status_code == 200
        assert 'public_models' in response.context
        assert public_model in response.context['public_models']

    def test_new_prediction_view_excludes_own_from_public(self, client, member_user, approved_model):
        """Test that user's own models are excluded from public section."""
        # Make the model public
        approved_model.is_public = True
        approved_model.save()

        client.force_login(member_user)
        url = reverse('inference:new_prediction')
        response = client.get(url)

        # Model should be in my_models, not public_models
        assert approved_model in response.context['my_models']
        assert approved_model not in response.context['public_models']

    def test_new_prediction_view_preselect_model(self, client, member_user, approved_model):
        """Test that model can be preselected via query param."""
        client.force_login(member_user)
        url = reverse('inference:new_prediction')
        response = client.get(url, {'model': approved_model.id})

        assert response.status_code == 200
        assert 'preselected_model' in response.context
        assert response.context['preselected_model'] == approved_model

    def test_new_prediction_view_filter_by_domain(self, client, member_user, approved_model, public_model):
        """Test filtering by domain."""
        client.force_login(member_user)
        url = reverse('inference:new_prediction')
        response = client.get(url, {'domain': 'cardiology'})

        assert response.status_code == 200
        assert response.context['domain_filter'] == 'cardiology'

    def test_new_prediction_view_search(self, client, member_user, approved_model):
        """Test search functionality."""
        client.force_login(member_user)
        url = reverse('inference:new_prediction')
        response = client.get(url, {'q': 'Test Prediction'})

        assert response.status_code == 200
        assert response.context['search_query'] == 'Test Prediction'


@pytest.mark.django_db
class TestPredictionLoadDataView:
    """Tests for the prediction_load_data view (Step 2)."""

    def test_load_data_view_requires_login(self, client):
        """Test that load_data view requires authentication."""
        url = reverse('inference:prediction_load_data')
        response = client.get(url)
        assert response.status_code == 302
        assert '/auth/login/' in response.url

    def test_load_data_view_requires_model_id(self, client, member_user):
        """Test that view redirects if no model_id provided."""
        client.force_login(member_user)
        url = reverse('inference:prediction_load_data')
        response = client.get(url)

        assert response.status_code == 302
        assert '/inference/predict/' in response.url  # Redirects to new_prediction

    def test_load_data_view_accepts_model_id_post(self, client, member_user, approved_model):
        """Test that view accepts model_id via POST."""
        client.force_login(member_user)
        url = reverse('inference:prediction_load_data')
        response = client.post(url, {'model_id': approved_model.id})

        assert response.status_code == 200
        assert 'inference/prediction_load_data.html' in [t.name for t in response.templates]

    def test_load_data_view_accepts_model_id_get(self, client, member_user, approved_model):
        """Test that view accepts model_id via GET."""
        client.force_login(member_user)
        url = reverse('inference:prediction_load_data')
        response = client.get(url, {'model_id': approved_model.id})

        assert response.status_code == 200

    def test_load_data_view_validates_model_access(self, client, member_user, public_model):
        """Test that user can access public models."""
        client.force_login(member_user)
        url = reverse('inference:prediction_load_data')
        response = client.post(url, {'model_id': public_model.id})

        assert response.status_code == 200
        assert response.context['model'] == public_model

    def test_load_data_view_rejects_invalid_model(self, client, member_user):
        """Test that invalid model_id is rejected."""
        client.force_login(member_user)
        url = reverse('inference:prediction_load_data')
        response = client.post(url, {'model_id': 99999})

        assert response.status_code == 302  # Redirects back

    def test_load_data_view_rejects_non_approved_model(self, client, member_user):
        """Test that pending models cannot be used."""
        # Create a pending model
        pending_model = DeployedModel.objects.create(
            name='Pending Model',
            version='1.0.0',
            description='Not approved yet',
            domain='cardiology',
            uploaded_by=member_user,
            status='pending',
            input_schema={'feature_names': []},
            output_schema={'type': 'regression'},
        )

        client.force_login(member_user)
        url = reverse('inference:prediction_load_data')
        response = client.post(url, {'model_id': pending_model.id})

        assert response.status_code == 302  # Redirects back

    def test_load_data_view_has_wizard_step_2(self, client, member_user, approved_model):
        """Test that view is at wizard step 2."""
        client.force_login(member_user)
        url = reverse('inference:prediction_load_data')
        response = client.post(url, {'model_id': approved_model.id})

        assert response.status_code == 200
        assert response.context['wizard_step'] == 2

    def test_load_data_view_has_model_context(self, client, member_user, approved_model):
        """Test that model is in context."""
        client.force_login(member_user)
        url = reverse('inference:prediction_load_data')
        response = client.post(url, {'model_id': approved_model.id})

        assert response.status_code == 200
        assert 'model' in response.context
        assert response.context['model'] == approved_model

    def test_load_data_view_breadcrumbs(self, client, member_user, approved_model):
        """Test that breadcrumbs are correct."""
        client.force_login(member_user)
        url = reverse('inference:prediction_load_data')
        response = client.post(url, {'model_id': approved_model.id})

        assert response.status_code == 200
        breadcrumbs = response.context['breadcrumbs']
        assert len(breadcrumbs) == 3
        assert breadcrumbs[1]['name'] == 'New Prediction'
        assert breadcrumbs[2]['name'] == 'Load Data'

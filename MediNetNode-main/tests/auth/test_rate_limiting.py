from django.test import TestCase, Client
from django.core.cache import cache
from users.models import CustomUser, Role


class RateLimitingTests(TestCase):
    def setUp(self) -> None:
        self.client = Client()
        role = Role.objects.get(name='ADMIN')
        CustomUser.objects.create_user(username='bob', password='StrongPass123!', role=role)
        cache.clear()

    def test_ip_rate_limit_blocks_after_5(self) -> None:
        for _ in range(5):
            res = self.client.post('/auth/login/', {'username': 'bob', 'password': 'wrong'})
            self.assertIn(res.status_code, (400, 403, 429))
        res = self.client.post('/auth/login/', {'username': 'bob', 'password': 'wrong'})
        self.assertIn(res.status_code, (403, 429))

    def test_user_rate_limit_blocks_after_10(self) -> None:
        for _ in range(10):
            res = self.client.post('/auth/login/', {'username': 'bob', 'password': 'wrong'}, HTTP_X_FORWARDED_FOR='203.0.113.10')
            self.assertIn(res.status_code, (400, 403, 429))
        res = self.client.post('/auth/login/', {'username': 'bob', 'password': 'wrong'}, HTTP_X_FORWARDED_FOR='203.0.113.10')
        self.assertIn(res.status_code, (403, 429))



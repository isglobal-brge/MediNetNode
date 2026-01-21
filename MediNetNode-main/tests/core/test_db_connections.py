from django.test import TestCase
from django.db import connections


class DBConnectionsTest(TestCase):
    databases = {"default", "datasets_db"}

    def test_databases_connections_ok(self) -> None:
        connections["default"].ensure_connection()
        connections["datasets_db"].ensure_connection()



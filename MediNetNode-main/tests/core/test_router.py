from core.routers import DatabaseRouter


class _Meta:
    def __init__(self, app_label: str) -> None:
        self.app_label = app_label


class _Model:
    def __init__(self, app_label: str) -> None:
        self._meta = _Meta(app_label)


def test_router_reads_and_writes_users_to_default() -> None:
    router = DatabaseRouter()
    model = _Model('users')
    assert router.db_for_read(model) == 'default'
    assert router.db_for_write(model) == 'default'


def test_router_reads_and_writes_datasets_to_datasets_db() -> None:
    router = DatabaseRouter()
    model = _Model('dataset')
    assert router.db_for_read(model) == 'datasets_db'
    assert router.db_for_write(model) == 'datasets_db'


def test_router_returns_none_for_unknown_apps() -> None:
    router = DatabaseRouter()
    model = _Model('unknown_app')
    assert router.db_for_read(model) is None
    assert router.db_for_write(model) is None



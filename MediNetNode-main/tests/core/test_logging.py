import logging
from pathlib import Path

from django.conf import settings


def test_logging_files_exist(tmp_path, monkeypatch) -> None:
    # Ensure logs dir exists and handlers can write
    logs_dir = Path(settings.BASE_DIR) / 'logs'
    logs_dir.mkdir(exist_ok=True)

    logger = logging.getLogger('django')
    logger.info('test log entry')

    # Security logger
    sec_logger = logging.getLogger('security')
    sec_logger.warning('security test')

    # Files should be present after first write
    assert (logs_dir / 'medinet.log').exists()
    assert (logs_dir / 'security.log').exists()



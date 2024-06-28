"""Global pytest configuration for the package."""

import imap_data_access
import pytest

# Import modularized fixtures as plugins from the fixtures package in this directory
pytest_plugins = [
    "imap_processing.tests.plugins.common_fixtures",
    "imap_processing.tests.plugins.hi_fixtures",
]


@pytest.fixture(autouse=True)
def _set_global_config(monkeypatch, tmp_path):
    """Set the global data directory to a temporary directory."""
    monkeypatch.setitem(imap_data_access.config, "DATA_DIR", tmp_path)
    monkeypatch.setitem(
        imap_data_access.config, "DATA_ACCESS_URL", "https://api.test.com"
    )

"""Global pytest configuration for the package."""
import pytest

import imap_processing


@pytest.fixture(autouse=True)
def _set_global_config(monkeypatch, tmp_path):
    """Set the global data directory to a temporary directory."""
    monkeypatch.setitem(imap_processing.config, "DATA_DIR", tmp_path)
    monkeypatch.setitem(imap_processing.config, "API_URL", "https://api.test.com")

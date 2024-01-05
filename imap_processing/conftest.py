"""Global pytest configuration for the package."""
import pytest

import imap_processing


@pytest.fixture(autouse=True)
def _set_global_data_dir(tmp_path):
    """Set the global data directory to a temporary directory."""
    _original_data_dir = imap_processing.config["DATA_DIR"]
    imap_processing.config["DATA_DIR"] = tmp_path
    yield
    imap_processing.config["DATA_DIR"] = _original_data_dir

"""Pytest fixtures that are not instrument specific."""

import pytest

from imap_processing import imap_module_directory


@pytest.fixture(scope="session")
def imap_tests_path():
    return imap_module_directory / "tests"

import pytest


@pytest.fixture(scope="session")
def swapi_test_data_path(imap_tests_path):
    return imap_tests_path / "swapi/"


@pytest.fixture(scope="session")
def swapi_l0_test_data_path(swapi_test_data_path):
    return swapi_test_data_path / "l0_data/"


@pytest.fixture(scope="session")
def swapi_l0_validation_data_path(swapi_test_data_path):
    return swapi_test_data_path / "l0_validation_data/"

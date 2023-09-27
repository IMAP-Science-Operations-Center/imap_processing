"""Pytest plugin module for test data paths"""
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def test_data_path():
    """Returns the Path to the test_data directory"""
    return Path(sys.modules[__name__.split(
        '.')[0]].__file__).parent / 'ultra' / 'tests' / 'test_data'


@pytest.fixture()
def ccsds_path(test_data_path):
    """Returns the ccsds directory.
    """
    return test_data_path / 'L0' / \
        'Ultra45_EM_SwRI_Cal_Run7_ThetaScan_20220530T225054.CCSDS'


@pytest.fixture()
def xtce_path():
    """Returns the xtce directory.
    """
    return Path(sys.modules[__name__.split(
        '.')[0]].__file__).parent / 'ultra' / 'packet_definitions'


@pytest.fixture()
def xtce_image_rates_path(xtce_path):
    """Returns the xtce image rates directory.
    """
    return xtce_path / "P_U45_IMAGE_RATES.xml"


@pytest.fixture()
def xtce_aux_path(xtce_path):
    """Returns the xtce auxilliary directory.
    """
    return xtce_path / "P_U45_AUXILIARY.xml"


@pytest.fixture()
def xtce_image_raw_events_path(xtce_path):
    """Returns the xtce image raw events directory.
    """
    return xtce_path / "P_U45_IMG_RAW_EVENTS.xml"


@pytest.fixture()
def xtce_image_rates_test_path(test_data_path):
    """Returns the xtce image rates test data directory.
    """
    filename = (
        "ultra45_raw_sc_ultraimgrates_Ultra45_EM_SwRI_Cal_Run7_ThetaScan_"
        "20220530T225054.csv"
    )
    return test_data_path / 'L0' / filename


@pytest.fixture()
def xtce_aux_test_path(test_data_path):
    """Returns the xtce auxiliary test data directory.
    """
    filename = (
        "ultra45_raw_sc_auxdata_Ultra45_EM_SwRI_Cal_Run7_ThetaScan_"
        "20220530T225054.csv"
    )
    return test_data_path / 'L0' / filename

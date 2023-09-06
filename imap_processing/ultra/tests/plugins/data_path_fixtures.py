"""Pytest plugin module for test data paths"""
# Standard
from pathlib import Path
import sys
# Installed
import pytest
from imap_processing import packet_definition_directory


@pytest.fixture
def test_data_path():
    """Returns the Path to the test_data directory"""
    return Path(sys.modules[__name__.split('.')[0]].__file__).parent / 'ultra'/ 'tests' / 'test_data'

@pytest.fixture
def ccsds_path(test_data_path):
    """Returns the spice subdirectory of the test_data directory
    This directory contains kernel that are either generated (SPK and CK) or dynamically downloaded.
    Any kernels that are available directly in the libera_utils/data directory should be sourced from there.
    """
    return test_data_path / 'L0' / 'Ultra45_EM_SwRI_Cal_Run7_ThetaScan_20220530T225054.CCSDS'

@pytest.fixture
def xtce_image_rates_path():
    """Returns the spice subdirectory of the test_data directory
    This directory contains kernel that are either generated (SPK and CK) or dynamically downloaded.
    Any kernels that are available directly in the libera_utils/data directory should be sourced from there.
    """
    return f"{packet_definition_directory}ultra/P_U45_IMAGE_RATES.xml"

@pytest.fixture
def xtce_aux_path():
    """Returns the spice subdirectory of the test_data directory
    This directory contains kernel that are either generated (SPK and CK) or dynamically downloaded.
    Any kernels that are available directly in the libera_utils/data directory should be sourced from there.
    """
    return f"{packet_definition_directory}ultra/P_U45_AUXILIARY.xml"

@pytest.fixture
def xtce_image_rates_test_path(test_data_path):
    """Returns the spice subdirectory of the test_data directory
    This directory contains kernel that are either generated (SPK and CK) or dynamically downloaded.
    Any kernels that are available directly in the libera_utils/data directory should be sourced from there.
    """
    return test_data_path / 'L0' / 'ultra45_raw_sc_ultraimgrates_Ultra45_EM_SwRI_Cal_Run7_ThetaScan_20220530T225054.csv'

@pytest.fixture
def xtce_aux_test_path(test_data_path):
    """Returns the spice subdirectory of the test_data directory
    This directory contains kernel that are either generated (SPK and CK) or dynamically downloaded.
    Any kernels that are available directly in the libera_utils/data directory should be sourced from there.
    """
    return test_data_path / 'L0' / 'ultra45_raw_sc_auxdata_Ultra45_EM_SwRI_Cal_Run7_ThetaScan_20220530T225054.csv'
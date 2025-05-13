"""Pytest plugin module for test data paths."""

import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.ultra.l0.decom_ultra import (
    process_ultra_events,
    process_ultra_rates,
    process_ultra_tof,
)
from imap_processing.ultra.l0.ultra_utils import (
    ULTRA_AUX,
    ULTRA_EVENTS,
    ULTRA_RATES,
    ULTRA_TOF,
)
from imap_processing.ultra.l1a.ultra_l1a import ultra_l1a
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture
def ccsds_path():
    """Returns the ccsds directory."""
    return (
        imap_module_directory
        / "tests"
        / "ultra"
        / "data"
        / "l0"
        / "Ultra45_EM_SwRI_Cal_Run7_ThetaScan_20220530T225054.CCSDS"
    )


@pytest.fixture
def ccsds_path_events():
    """Returns the ccsds directory."""
    return (
        imap_module_directory
        / "tests"
        / "ultra"
        / "data"
        / "l0"
        / "FM45_7P_Phi0.0_BeamCal_LinearScan_phi0.04_theta-0.01_20230821T121304.CCSDS"
    )


@pytest.fixture
def ccsds_path_theta_0():
    """Returns the ccsds directory."""
    return (
        imap_module_directory
        / "tests"
        / "ultra"
        / "data"
        / "l0"
        / "FM45_40P_Phi28p5_BeamCal_LinearScan_phi28.50_theta-0.00"
        "_20240207T102740.CCSDS"
    )


@pytest.fixture
def ccsds_path_tof():
    """Returns the ccsds directory."""
    return (
        imap_module_directory
        / "tests"
        / "ultra"
        / "data"
        / "l0"
        / "FM45_TV_Cycle6_Hot_Ops_Front212_20240124T063837.CCSDS"
    )


@pytest.fixture
def xtce_path():
    """Returns the xtce image rates directory."""
    return (
        imap_module_directory
        / "ultra"
        / "packet_definitions"
        / "ULTRA_SCI_COMBINED.xml"
    )


@pytest.fixture
def rates_test_path():
    """Returns the xtce image rates test data directory."""
    filename = (
        "ultra45_raw_sc_ultraimgrates_Ultra45_EM_SwRI_Cal_Run7_ThetaScan_"
        "20220530T225054.csv"
    )
    return imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename


@pytest.fixture
def aux_test_path():
    """Returns the xtce auxiliary test data directory."""
    filename = (
        "ultra45_raw_sc_auxdata_Ultra45_EM_SwRI_Cal_Run7_ThetaScan_20220530T225054.csv"
    )
    return imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename


@pytest.fixture
def events_test_path():
    """Returns the xtce auxiliary test data directory."""
    filename = (
        "ultra45_raw_sc_ultrarawimgevent_FM45_7P_Phi00_BeamCal_"
        "LinearScan_phi004_theta-001_20230821T121304.csv"
    )
    return imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename


@pytest.fixture
def tof_test_path():
    """Returns the xtce auxiliary test data directory."""
    filename = (
        "ultra45_raw_sc_enaphxtofhangimg_FM45_TV_Cycle6_Hot_Ops_"
        "Front212_20240124T063837.csv"
    )
    return imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename


@pytest.fixture
def decom_test_data(request, xtce_path):
    """Read test data from file"""
    apid = request.param["apid"]
    filename = request.param["filename"]
    ccsds_path = imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename

    datasets_by_apid = packet_file_to_datasets(ccsds_path, xtce_path)

    strategy_dict = {
        ULTRA_TOF.apid[0]: process_ultra_tof,
        ULTRA_EVENTS.apid[0]: process_ultra_events,
        ULTRA_RATES.apid[0]: process_ultra_rates,
        ULTRA_TOF.apid[1]: process_ultra_tof,
        ULTRA_EVENTS.apid[1]: process_ultra_events,
        ULTRA_RATES.apid[1]: process_ultra_rates,
    }

    process_function = strategy_dict.get(apid, lambda *args: False)
    data_packet_xarray = process_function(datasets_by_apid[apid])

    return data_packet_xarray


@pytest.fixture
def events_fsw_comparison_theta_0():
    """FSW test data."""
    filename = (
        "ultra45_raw_sc_ultrarawimg_withFSWcalcs_FM45_40P_Phi28p5_"
        "BeamCal_LinearScan_phi2850_theta-000_20240207T102740.csv"
    )
    return imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename


@pytest.fixture
def de_dataset(ccsds_path_theta_0, xtce_path):
    """L1A test data"""
    test_data = ultra_l1a(ccsds_path_theta_0, apid_input=ULTRA_EVENTS.apid[0])
    return test_data[0]


@pytest.fixture
def rates_dataset(ccsds_path_theta_0):
    """L1A test data"""
    test_data = ultra_l1a(ccsds_path_theta_0, apid_input=ULTRA_RATES.apid[0])
    return test_data[0]


@pytest.fixture
def aux_dataset(ccsds_path_theta_0):
    """L1A test data"""
    test_data = ultra_l1a(ccsds_path_theta_0, apid_input=ULTRA_AUX.apid[0])
    return test_data[0]


@pytest.fixture
def faux_aux_dataset():
    """Fixture to compute and return aux test data."""

    num_spins = 15
    spin_duration = 15  # in seconds

    epoch = np.arange(0, num_spins, 1)
    spin_number = np.arange(127, 142)
    spin_start_time = np.arange(1905, 2115 + spin_duration, spin_duration)
    spin_period_sec = np.full(num_spins, 15)
    spin_period_sec[-1] = 14
    spin_start_sec = np.arange(1905, 2130, 15)
    spin_start_subsec = np.zeros(num_spins)

    test_aux_dataset = xr.Dataset(
        data_vars={
            "TIMESPINSTART": ("epoch", spin_start_sec),
            "TIMESPINSTARTSUB": ("epoch", spin_start_subsec),
            "DURATION": ("epoch", spin_period_sec),
            "SPINNUMBER": ("epoch", spin_number),
            "TIMESPINDATA": ("epoch", spin_start_time),
            "SPINPERIOD": ("epoch", spin_period_sec),
        },
        coords={"epoch": ("epoch", epoch)},
    )

    return test_aux_dataset

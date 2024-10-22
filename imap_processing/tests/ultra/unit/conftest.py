"""Pytest plugin module for test data paths"""

import numpy as np
import pytest

from imap_processing import decom, imap_module_directory
from imap_processing.ultra.l0.decom_ultra import process_ultra_apids
from imap_processing.ultra.l0.ultra_utils import (
    ULTRA_AUX,
    ULTRA_EVENTS,
)
from imap_processing.ultra.l1a import ultra_l1a
from imap_processing.utils import group_by_apid


@pytest.fixture()
def ccsds_path():
    """Returns the ccsds directory."""
    return (
        imap_module_directory
        / "tests"
        / "ultra"
        / "test_data"
        / "l0"
        / "Ultra45_EM_SwRI_Cal_Run7_ThetaScan_20220530T225054.CCSDS"
    )


@pytest.fixture()
def ccsds_path_events():
    """Returns the ccsds directory."""
    return (
        imap_module_directory
        / "tests"
        / "ultra"
        / "test_data"
        / "l0"
        / "FM45_7P_Phi0.0_BeamCal_LinearScan_phi0.04_theta-0.01_20230821T121304.CCSDS"
    )


@pytest.fixture()
def ccsds_path_theta_0():
    """Returns the ccsds directory."""
    return (
        imap_module_directory
        / "tests"
        / "ultra"
        / "test_data"
        / "l0"
        / "FM45_40P_Phi28p5_BeamCal_LinearScan_phi28.50_theta-0.00"
        "_20240207T102740.CCSDS"
    )


@pytest.fixture()
def ccsds_path_tof():
    """Returns the ccsds directory."""
    return (
        imap_module_directory
        / "tests"
        / "ultra"
        / "test_data"
        / "l0"
        / "FM45_TV_Cycle6_Hot_Ops_Front212_20240124T063837.CCSDS"
    )


@pytest.fixture()
def xtce_path():
    """Returns the xtce image rates directory."""
    return (
        imap_module_directory
        / "ultra"
        / "packet_definitions"
        / "ULTRA_SCI_COMBINED.xml"
    )


@pytest.fixture()
def rates_test_path():
    """Returns the xtce image rates test data directory."""
    filename = (
        "ultra45_raw_sc_ultraimgrates_Ultra45_EM_SwRI_Cal_Run7_ThetaScan_"
        "20220530T225054.csv"
    )
    return imap_module_directory / "tests" / "ultra" / "test_data" / "l0" / filename


@pytest.fixture()
def aux_test_path():
    """Returns the xtce auxiliary test data directory."""
    filename = (
        "ultra45_raw_sc_auxdata_Ultra45_EM_SwRI_Cal_Run7_ThetaScan_"
        "20220530T225054.csv"
    )
    return imap_module_directory / "tests" / "ultra" / "test_data" / "l0" / filename


@pytest.fixture()
def events_test_path():
    """Returns the xtce auxiliary test data directory."""
    filename = (
        "ultra45_raw_sc_ultrarawimgevent_FM45_7P_Phi00_BeamCal_"
        "LinearScan_phi004_theta-001_20230821T121304.csv"
    )
    return imap_module_directory / "tests" / "ultra" / "test_data" / "l0" / filename


@pytest.fixture()
def tof_test_path():
    """Returns the xtce auxiliary test data directory."""
    filename = (
        "ultra45_raw_sc_enaphxtofhangimg_FM45_TV_Cycle6_Hot_Ops_"
        "Front212_20240124T063837.csv"
    )
    return imap_module_directory / "tests" / "ultra" / "test_data" / "l0" / filename


@pytest.fixture()
def decom_test_data(request, xtce_path):
    """Read test data from file"""
    apid = request.param["apid"]
    filename = request.param["filename"]
    ccsds_path = (
        imap_module_directory / "tests" / "ultra" / "test_data" / "l0" / filename
    )

    packets = decom.decom_packets(ccsds_path, xtce_path)
    grouped_data = group_by_apid(packets)

    data_packet_list = process_ultra_apids(grouped_data[apid], apid)
    return data_packet_list, packets


@pytest.fixture()
def events_fsw_comparison_theta_0():
    """FSW test data."""
    filename = (
        "ultra45_raw_sc_ultrarawimg_withFSWcalcs_FM45_40P_Phi28p5_"
        "BeamCal_LinearScan_phi2850_theta-000_20240207T102740.csv"
    )
    return imap_module_directory / "tests" / "ultra" / "test_data" / "l0" / filename


@pytest.fixture()
def de_dataset(ccsds_path_theta_0, xtce_path):
    """L1A test data"""
    packets = decom.decom_packets(ccsds_path_theta_0, xtce_path)
    grouped_data = group_by_apid(packets)
    decom_ultra_events = process_ultra_apids(
        grouped_data[ULTRA_EVENTS.apid[0]], ULTRA_EVENTS.apid[0]
    )
    decom_ultra_aux = process_ultra_apids(
        grouped_data[ULTRA_AUX.apid[0]], ULTRA_AUX.apid[0]
    )
    dataset = ultra_l1a.create_dataset(
        {
            ULTRA_EVENTS.apid[0]: decom_ultra_events,
            ULTRA_AUX.apid[0]: decom_ultra_aux,
        }
    )
    # Remove start_type with fill values
    l1a_de_dataset = dataset.where(
        dataset["START_TYPE"] != np.iinfo(np.int64).min, drop=True
    )
    return l1a_de_dataset

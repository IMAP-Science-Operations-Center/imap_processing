"""Pytest plugin module for test data paths"""

from unittest import mock

import numpy as np
import pytest
import xarray as xr

from imap_processing import decom, imap_module_directory
from imap_processing.ultra.l0.decom_ultra import process_ultra_apids
from imap_processing.ultra.l0.ultra_utils import (
    ULTRA_AUX,
    ULTRA_EVENTS,
    ULTRA_RATES,
)
from imap_processing.ultra.l1a import ultra_l1a
from imap_processing.ultra.l1b.ultra_l1b import ultra_l1b
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
    de_dataset = ultra_l1a.create_dataset(
        {
            ULTRA_EVENTS.apid[0]: decom_ultra_events,
            ULTRA_AUX.apid[0]: decom_ultra_aux,
        }
    )

    return de_dataset


@pytest.fixture()
def rates_dataset(ccsds_path_theta_0, xtce_path):
    """L1A test data"""
    packets = decom.decom_packets(ccsds_path_theta_0, xtce_path)
    grouped_data = group_by_apid(packets)
    decom_ultra_rates = process_ultra_apids(
        grouped_data[ULTRA_RATES.apid[0]], ULTRA_RATES.apid[0]
    )
    l1a_rates_dataset = ultra_l1a.create_dataset(
        {
            ULTRA_RATES.apid[0]: decom_ultra_rates,
        }
    )
    return l1a_rates_dataset


@pytest.fixture()
def aux_dataset(ccsds_path_theta_0, xtce_path):
    """L1A test data"""
    packets = decom.decom_packets(ccsds_path_theta_0, xtce_path)
    grouped_data = group_by_apid(packets)
    decom_ultra_aux = process_ultra_apids(
        grouped_data[ULTRA_AUX.apid[0]], ULTRA_AUX.apid[0]
    )
    l1a_aux_dataset = ultra_l1a.create_dataset(
        {
            ULTRA_AUX.apid[0]: decom_ultra_aux,
        }
    )
    return l1a_aux_dataset


@pytest.fixture()
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


@pytest.fixture()
@mock.patch("imap_processing.ultra.l1b.de.get_annotated_particle_velocity")
def l1b_de_dataset(
    mock_get_annotated_particle_velocity,
    de_dataset,
    use_fake_spin_data_for_time,
):
    """L1B test data"""

    data_dict = {}
    data_dict[de_dataset.attrs["Logical_source"]] = de_dataset
    # Create a spin table that cover spin 0-141
    use_fake_spin_data_for_time(0, 141 * 15)

    # Mock get_annotated_particle_velocity to avoid needing kernels
    def side_effect_func(event_times, position, ultra_frame, dps_frame, sc_frame):
        """
        Mock behavior of get_annotated_particle_velocity.

        Returns NaN-filled arrays matching the expected output shape.
        """
        num_events = event_times.size
        return (
            np.full((num_events, 3), np.nan),  # sc_velocity
            np.full((num_events, 3), np.nan),  # sc_dps_velocity
            np.full((num_events, 3), np.nan),  # helio_velocity
        )

    mock_get_annotated_particle_velocity.side_effect = side_effect_func

    output_datasets = ultra_l1b(data_dict, data_version="001")

    return output_datasets


@pytest.fixture()
def l1b_extendedspin_dataset(
    l1b_de_dataset,
    rates_dataset,
    faux_aux_dataset,
):
    """L1B de test data"""
    data_dict = {}
    data_dict["imap_ultra_l1b_45sensor-de"] = l1b_de_dataset[0]
    data_dict["imap_ultra_l1a_45sensor-aux"] = faux_aux_dataset
    # TODO: this is a placeholder for the hk dataset.
    data_dict["imap_ultra_l1a_45sensor-hk"] = faux_aux_dataset
    data_dict["imap_ultra_l1a_45sensor-rates"] = rates_dataset

    output_datasets = ultra_l1b(data_dict, data_version="001")

    return output_datasets

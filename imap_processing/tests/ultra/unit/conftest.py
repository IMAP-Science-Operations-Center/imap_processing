"""Pytest plugin module for test data paths"""
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def ccsds_path():
    """Returns the ccsds directory."""
    return (
        Path(sys.modules[__name__.split(".")[0]].__file__).parent
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
        Path(sys.modules[__name__.split(".")[0]].__file__).parent
        / "tests"
        / "ultra"
        / "test_data"
        / "l0"
        / "FM45_7P_Phi0.0_BeamCal_LinearScan_phi0.04_theta-0.01_20230821T121304.CCSDS"
    )


@pytest.fixture()
def ccsds_path_tof():
    """Returns the ccsds directory."""
    return (
        Path(sys.modules[__name__.split(".")[0]].__file__).parent
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
        Path(sys.modules[__name__.split(".")[0]].__file__).parent
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
    return (
        Path(sys.modules[__name__.split(".")[0]].__file__).parent
        / "tests"
        / "ultra"
        / "test_data"
        / "l0"
        / filename
    )


@pytest.fixture()
def aux_test_path():
    """Returns the xtce auxiliary test data directory."""
    filename = (
        "ultra45_raw_sc_auxdata_Ultra45_EM_SwRI_Cal_Run7_ThetaScan_"
        "20220530T225054.csv"
    )
    return (
        Path(sys.modules[__name__.split(".")[0]].__file__).parent
        / "tests"
        / "ultra"
        / "test_data"
        / "l0"
        / filename
    )


@pytest.fixture()
def events_test_path():
    """Returns the xtce auxiliary test data directory."""
    filename = (
        "ultra45_raw_sc_ultrarawimgevent_FM45_7P_Phi00_BeamCal_"
        "LinearScan_phi004_theta-001_20230821T121304.csv"
    )
    return (
        Path(sys.modules[__name__.split(".")[0]].__file__).parent
        / "tests"
        / "ultra"
        / "test_data"
        / "l0"
        / filename
    )


@pytest.fixture()
def tof_test_path():
    """Returns the xtce auxiliary test data directory."""
    filename = (
        "ultra45_raw_sc_enaphxtofhangimg_FM45_TV_Cycle6_Hot_Ops_"
        "Front212_20240124T063837.csv"
    )
    return (
        Path(sys.modules[__name__.split(".")[0]].__file__).parent
        / "tests"
        / "ultra"
        / "test_data"
        / "l0"
        / filename
    )

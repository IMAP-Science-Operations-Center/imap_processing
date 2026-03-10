"""Pytest plugin module for test data paths."""

from unittest import mock

import astropy_healpix.healpy as hp
import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.ultra.l0.decom_ultra import (
    process_ultra_cmd_echo,
    process_ultra_energy_rates,
    process_ultra_energy_spectra,
    process_ultra_events,
    process_ultra_macros_checksum,
    process_ultra_rates,
    process_ultra_tof,
)
from imap_processing.ultra.l0.ultra_utils import (
    ULTRA_AUX,
    ULTRA_CMD_ECHO,
    ULTRA_ENERGY_EVENTS,
    ULTRA_ENERGY_RATES,
    ULTRA_ENERGY_SPECTRA,
    ULTRA_EVENTS,
    ULTRA_EXTOF_HIGH_ANGULAR,
    ULTRA_EXTOF_HIGH_ENERGY,
    ULTRA_EXTOF_HIGH_TIME,
    ULTRA_HK,
    ULTRA_MACROS_CHECKSUM,
    ULTRA_PHXTOF_HIGH_ANGULAR,
    ULTRA_PHXTOF_HIGH_ENERGY,
    ULTRA_PHXTOF_HIGH_TIME,
    ULTRA_PRI_1_EVENTS,
    ULTRA_PRI_2_EVENTS,
    ULTRA_PRI_3_EVENTS,
    ULTRA_PRI_4_EVENTS,
    ULTRA_RATES,
)
from imap_processing.ultra.l1a.ultra_l1a import ultra_l1a
from imap_processing.ultra.l1b.ultra_l1b_culling import (
    get_binned_energy_ranges,
    get_energy_range_flags,
)
from imap_processing.ultra.l1c.l1c_lookup_utils import build_energy_bins
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
def ccsds_path_all_apids():
    """Returns the ccsds directory."""
    return (
        imap_module_directory
        / "tests"
        / "ultra"
        / "data"
        / "l0"
        / "imap_ultra_l0_raw_20260924_v001.pkts"
    )


@pytest.fixture
def ccsds_path_tof_high_angular():
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
def ccsds_path_functional():
    """Returns the ccsds directory."""
    return (
        imap_module_directory
        / "tests"
        / "ultra"
        / "data"
        / "l0"
        / "FM45_UltraFM45_Functional_2024-01-22T0105_20240122T010548.CCSDS"
    )


@pytest.fixture
def ccsds_path_startup():
    """Returns the ccsds directory."""
    return (
        imap_module_directory
        / "tests"
        / "ultra"
        / "data"
        / "l0"
        / "FM90_Startup_20230711T081655.CCSDS"
    )


@pytest.fixture
def ccsds_path_extra():
    """Returns the ccsds directory."""
    return (
        imap_module_directory
        / "tests"
        / "ultra"
        / "data"
        / "l0"
        / "FM45_UltraFM45Extra_TV_Tests_2024-01-22T0930_20240122T093008.CCSDS"
    )


@pytest.fixture
def xtce_path():
    """Returns the xtce directory."""
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
def rates_l1_test_path():
    filename = (
        "FM45_40P_Phi28p5_BeamCal_LinearScan_phi28.50_theta-0.00_"
        "ULTRA_ImageBasicRates_20240207T102740_.csv"
    )
    return imap_module_directory / "tests" / "ultra" / "data" / "l1" / filename


@pytest.fixture
def energy_rates_test_path():
    """Returns the xtce test data directory."""
    filename = (
        "ultra45_raw_sc_ultranrgrates_FM45_UltraFM45_Functional"
        "_2024-01-22T0105_20240122T010548.csv"
    )
    return imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename


@pytest.fixture
def energy_spectra_test_path():
    """Returns the xtce test data directory."""
    filename = "ultra90_raw_sc_ultraenergyspctr_FM90_Startup_20230711T081655.csv"
    return imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename


@pytest.fixture
def priority_1_test_path():
    """Returns the xtce test data directory."""
    filename = (
        "ultra45_raw_sc_imgpriority1evnt_FM45_UltraFM45Extra_TV_Tests_"
        "2024-01-22T0930_20240122T093008.csv"
    )
    return imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename


@pytest.fixture
def priority_2_test_path():
    """Returns the xtce test data directory."""
    filename = (
        "ultra45_raw_sc_imgpriority2evnt_FM45_UltraFM45Extra_TV_Tests_"
        "2024-01-22T0930_20240122T093008.csv"
    )
    return imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename


@pytest.fixture
def priority_3_test_path():
    """Returns the xtce test data directory."""
    filename = (
        "ultra45_raw_sc_imgpriority3evnt_FM45_UltraFM45Extra_TV_Tests_"
        "2024-01-22T0930_20240122T093008.csv"
    )
    return imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename


@pytest.fixture
def priority_4_test_path():
    """Returns the xtce test data directory."""
    filename = (
        "ultra45_raw_sc_imgpriority4evnt_FM45_UltraFM45Extra_TV_Tests_"
        "2024-01-22T0930_20240122T093008.csv"
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
    """Returns the xtce test data directory."""
    filename = (
        "ultra45_raw_sc_ultrarawimgevent_FM45_7P_Phi00_BeamCal_"
        "LinearScan_phi004_theta-001_20230821T121304.csv"
    )
    return imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename


@pytest.fixture
def tof_high_angular_test_path():
    """Returns the xtce test data directory."""
    filename = (
        "ultra45_raw_sc_enaphxtofhangimg_FM45_TV_Cycle6_Hot_Ops_"
        "Front212_20240124T063837.csv"
    )
    return imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename


@pytest.fixture
def tof_high_energy_test_path():
    """Returns the xtce test data directory."""
    filename = "ultra45_l1b_raw_sc_enaphxtofhnrgimg_20240122_00_SDCStyle.csv"
    return imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename


@pytest.fixture
def tof_high_time_test_path():
    """Returns the xtce test data directory."""
    filename = "ultra45_l1b_raw_sc_enaphxtofhtimeimg_20240122_00_SDCStyle.csv"
    return imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename


@pytest.fixture
def extof_high_angular_test_path():
    """Returns the xtce test data directory."""
    filename = "ultra45_l1b_raw_sc_enaextofhangimg_20240122_00_SDCStyle.csv"
    return imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename


@pytest.fixture
def extof_high_time_test_path():
    """Returns the xtce test data directory."""
    filename = "ultra45_l1b_raw_sc_ionextofhtimeimg_20240122_00_SDCStyle.csv"
    return imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename


@pytest.fixture
def extof_high_energy_test_path():
    """Returns the xtce test data directory."""
    filename = "ultra45_l1b_raw_sc_ionextofhnrgimg_20240122_00_SDCStyle.csv"
    return imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename


@pytest.fixture
def cmd_echo_test_path():
    """Returns the xtce test data directory."""
    filename = (
        "ultra45_raw_hk_ultracmdecho_FM45_UltraFM45_Functional_"
        "2024-01-22T0105_20240122T010548.csv"
    )
    return imap_module_directory / "tests" / "ultra" / "data" / "l0" / filename


@pytest.fixture
def macrochecksum_test_path():
    """Returns the xtce auxiliary test data directory."""
    filename = (
        "ultra45_raw_hk_macrochecksumrpt_FM45_UltraFM45_Functional_"
        "2024-01-22T0105_20240122T010548.csv"
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
        ULTRA_PHXTOF_HIGH_ANGULAR.apid[0]: lambda ds, apid: process_ultra_tof(
            ds, ULTRA_PHXTOF_HIGH_ANGULAR
        ),
        ULTRA_PHXTOF_HIGH_ANGULAR.apid[1]: lambda ds, apid: process_ultra_tof(
            ds, ULTRA_PHXTOF_HIGH_ANGULAR
        ),
        ULTRA_PHXTOF_HIGH_ENERGY.apid[0]: lambda ds, apid: process_ultra_tof(
            ds, ULTRA_PHXTOF_HIGH_ENERGY
        ),
        ULTRA_PHXTOF_HIGH_ENERGY.apid[1]: lambda ds, apid: process_ultra_tof(
            ds, ULTRA_PHXTOF_HIGH_ENERGY
        ),
        ULTRA_PHXTOF_HIGH_TIME.apid[0]: lambda ds, apid: process_ultra_tof(
            ds, ULTRA_PHXTOF_HIGH_TIME
        ),
        ULTRA_PHXTOF_HIGH_TIME.apid[1]: lambda ds, apid: process_ultra_tof(
            ds, ULTRA_PHXTOF_HIGH_TIME
        ),
        ULTRA_EXTOF_HIGH_ANGULAR.apid[0]: lambda ds, apid: process_ultra_tof(
            ds, ULTRA_EXTOF_HIGH_ANGULAR
        ),
        ULTRA_EXTOF_HIGH_ANGULAR.apid[1]: lambda ds, apid: process_ultra_tof(
            ds, ULTRA_EXTOF_HIGH_ANGULAR
        ),
        ULTRA_EXTOF_HIGH_TIME.apid[0]: lambda ds, apid: process_ultra_tof(
            ds, ULTRA_EXTOF_HIGH_TIME
        ),
        ULTRA_EXTOF_HIGH_TIME.apid[1]: lambda ds, apid: process_ultra_tof(
            ds, ULTRA_EXTOF_HIGH_TIME
        ),
        ULTRA_EXTOF_HIGH_ENERGY.apid[0]: lambda ds, apid: process_ultra_tof(
            ds, ULTRA_EXTOF_HIGH_ENERGY
        ),
        ULTRA_EXTOF_HIGH_ENERGY.apid[1]: lambda ds, apid: process_ultra_tof(
            ds, ULTRA_EXTOF_HIGH_ENERGY
        ),
        ULTRA_ENERGY_EVENTS.apid[0]: lambda ds, apid: process_ultra_events(ds, apid),
        ULTRA_ENERGY_EVENTS.apid[1]: lambda ds, apid: process_ultra_events(ds, apid),
        ULTRA_EVENTS.apid[0]: lambda ds, apid: process_ultra_events(ds, apid),
        ULTRA_EVENTS.apid[1]: lambda ds, apid: process_ultra_events(ds, apid),
        ULTRA_MACROS_CHECKSUM.apid[0]: lambda ds, apid: process_ultra_macros_checksum(
            ds
        ),
        ULTRA_MACROS_CHECKSUM.apid[1]: lambda ds, apid: process_ultra_macros_checksum(
            ds
        ),
        ULTRA_PRI_1_EVENTS.apid[0]: lambda ds, apid: process_ultra_events(ds, apid),
        ULTRA_PRI_1_EVENTS.apid[1]: lambda ds, apid: process_ultra_events(ds, apid),
        ULTRA_PRI_2_EVENTS.apid[0]: lambda ds, apid: process_ultra_events(ds, apid),
        ULTRA_PRI_2_EVENTS.apid[1]: lambda ds, apid: process_ultra_events(ds, apid),
        ULTRA_PRI_3_EVENTS.apid[0]: lambda ds, apid: process_ultra_events(ds, apid),
        ULTRA_PRI_3_EVENTS.apid[1]: lambda ds, apid: process_ultra_events(ds, apid),
        ULTRA_PRI_4_EVENTS.apid[0]: lambda ds, apid: process_ultra_events(ds, apid),
        ULTRA_PRI_4_EVENTS.apid[1]: lambda ds, apid: process_ultra_events(ds, apid),
        ULTRA_RATES.apid[0]: lambda ds, apid: process_ultra_rates(ds),
        ULTRA_RATES.apid[1]: lambda ds, apid: process_ultra_rates(ds),
        ULTRA_ENERGY_RATES.apid[0]: lambda ds, apid: process_ultra_energy_rates(ds),
        ULTRA_ENERGY_RATES.apid[1]: lambda ds, apid: process_ultra_energy_rates(ds),
        ULTRA_ENERGY_SPECTRA.apid[0]: lambda ds, apid: process_ultra_energy_spectra(ds),
        ULTRA_ENERGY_SPECTRA.apid[1]: lambda ds, apid: process_ultra_energy_spectra(ds),
        ULTRA_CMD_ECHO.apid[0]: lambda ds, apid: process_ultra_cmd_echo(ds),
        ULTRA_CMD_ECHO.apid[1]: lambda ds, apid: process_ultra_cmd_echo(ds),
    }

    process_function = strategy_dict.get(apid, lambda *args: False)
    data_packet_xarray = process_function(datasets_by_apid[apid], apid)

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
def events_fsw_comparison_theta_0_revised():
    """FSW test data."""
    filename = (
        "ultra45_raw_sc_ultrarawimg_withFSWccs_FM45_40P_Phi28p5_"
        "BeamCal_LinearScan_phi2850_theta-000_20240207T102740_revised20250724.csv"
    )
    return imap_module_directory / "tests" / "ultra" / "data" / "l1" / filename


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
def status_dataset(ccsds_path_theta_0):
    """L1A test data"""
    test_data = ultra_l1a(ccsds_path_theta_0, apid_input=ULTRA_HK.apid[3])
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
            "timespinstart": ("epoch", spin_start_sec),
            "timespinstartsub": ("epoch", spin_start_subsec),
            "duration": ("epoch", spin_period_sec),
            "spinnumber": ("epoch", spin_number),
            "timespindata": ("epoch", spin_start_time),
            "spinperiod": ("epoch", spin_period_sec),
        },
        coords={"epoch": ("epoch", epoch)},
    )

    return test_aux_dataset


@pytest.fixture
def ancillary_files():
    """Fixture to return ancillary files."""
    path = imap_module_directory / "tests" / "ultra" / "data" / "l1"
    return {
        "l1b-45sensor-logistic-interpolation": path
        / "imap_ultra_l1b-45sensor-logistic-interpolation_20250101_v000.csv",
        "l1b-sensor-gf-noblades": path
        / "imap_ultra_l1b-sensor-gf-noblades_20250101_v000.csv",
        "l1b-sensor-gf-blades": path
        / "imap_ultra_l1b-sensor-gf-blades_20250101_v000.csv",
        "l1b-45sensor-leftslit-lookup": path
        / "imap_ultra_l1b-45sensor-leftslit-lookup_20250101_v000.csv",
        "l1b-45sensor-rightslit-lookup": path
        / "imap_ultra_l1b-45sensor-rightslit-lookup_20250101_v000.csv",
        "l1b-45sensor-imgparams-lookup": path
        / "imap_ultra_l1b-45sensor-imgparams-lookup_20250101_v001.csv",
        "l1b-90sensor-imgparams-lookup": path
        / "imap_ultra_l1b-90sensor-imgparams-lookup_20250101_v001.csv",
        "l1b-45sensor-tdc-norm-lookup": path
        / "imap_ultra_l1b-45sensor-tdc-norm-lookup_20250101_v000.csv",
        "l1b-45sensor-back-pos-lookup": path
        / "imap_ultra_l1b-45sensor-back-pos-lookup_20250101_v000.csv",
        "l1b-egynorm-lookup": path / "imap_ultra_l1b-egynorm-lookup_20250101_v000.csv",
        "l1b-yadjust-lookup": path / "imap_ultra_l1b-yadjust-lookup_20250101_v001.csv",
        "l1b-45sensor-sptpphcorr": path
        / "imap_ultra_l1b-45sensor-sptpphcorr_20250101_v000.csv",
        "l1b-45sensor-spbtphcorr": path
        / "imap_ultra_l1b-45sensor-spbtphcorr_20250101_v000.csv",
        "l1b-90sensor-sptpphcorr": path
        / "imap_ultra_l1b-90sensor-sptpphcorr_20250101_v000.csv",
        "l1b-90sensor-spbtphcorr": path
        / "imap_ultra_l1b-90sensor-spbtphcorr_20250101_v000.csv",
        "l1b-45sensor-tofxeflat": path
        / "imap_ultra_l1b-45sensor-tofxeflat_20250101_v000.pgm",
        "l1b-45sensor-tofxemedium": path
        / "imap_ultra_l1b-45sensor-tofxemedium_20250101_v000.pgm",
        "l1b-45sensor-tofxesteep": path
        / "imap_ultra_l1b-45sensor-tofxesteep_20250101_v000.pgm",
        "l1b-90sensor-tofxeflat": path
        / "imap_ultra_l1b-90sensor-tofxeflat_20250101_v000.pgm",
        "l1b-90sensor-tofxemediu": path
        / "imap_ultra_l1b-90sensor-tofxemedium_20250101_v000.pgm",
        "l1b-90sensor-tofxesteep": path
        / "imap_ultra_l1b-90sensor-tofxesteep_20250101_v000.pgm",
        "l1b-tofxph": path / "imap_ultra_l1b-tofxph_20250101_v000.pgm",
        "l1b-90sensor-scattering-calibration-data": path
        / "imap_ultra_l1b-90sensor-scattering-calibration-data_20250101_v000.csv",
        "l1c-90sensor-dps-exposure": path
        / "imap_ultra_l1c-90sensor-dps-exposure_20250101_v000.csv",
        "l1c-90sensor-efficiencies": path
        / "imap_ultra_l1c-90sensor-efficiencies_20250101_v000.csv",
        "l1c-90sensor-gf": path / "imap_ultra_l1c-90sensor-gf_20250101_v000.csv",
        "l1c-90sensor-sc-pointing-theta": path
        / "imap_ultra_l1c-90sensor-sc-pointing-theta-test_20250101_v000.csv",
        "l1c-90sensor-sc-pointing-phi": path
        / "imap_ultra_l1c-90sensor-sc-pointing-phi-test_20250101_v000.csv",
        "l1c-90sensor-sc-pointing-index": path
        / "imap_ultra_l1c-90sensor-sc-pointing-index-test_20250101_v000.csv",
        "l1c-90sensor-sc-pointing-bsf": path
        / "imap_ultra_l1c-90sensor-sc-pointing-bsf-test_20250101_v000.csv",
        "l1c-45sensor-sc-pointing-theta": path
        / "imap_ultra_l1c-90sensor-sc-pointing-theta-test_20250101_v000.csv",
        "l1c-45sensor-sc-pointing-phi": path
        / "imap_ultra_l1c-90sensor-sc-pointing-phi-test_20250101_v000.csv",
        "l1c-45sensor-sc-pointing-index": path
        / "imap_ultra_l1c-90sensor-sc-pointing-index-test_20250101_v000.csv",
        "l1c-45sensor-sc-pointing-bsf": path
        / "imap_ultra_l1c-90sensor-sc-pointing-bsf-test_20250101_v000.csv",
        "l1b-scattering-thresholds-per-energy": path
        / "imap_ultra_l1b-scattering-thresholds-per-energy_20250101_v000.csv",
        "l1c-45sensor-static-dead-times": path
        / "imap_ultra_l1c-45sensor-static-dead-times_20250101_v000.csv",
        "l1c-90sensor-static-dead-times": path
        / "imap_ultra_l1c-90sensor-static-dead-times_20250101_v000.csv",
    }


@pytest.fixture
def mock_spacecraft_pointing_lookups():
    """Test lookup tables fixture."""
    np.random.seed(42)
    pix = hp.nside2npix(128)  # reduced for testing
    steps = 2  # Reduced for testing
    for_indices_by_spin_phase = xr.DataArray(
        np.random.choice([True, False], size=(steps, pix), p=[0.1, 0.9]),
        dims=("spin_phase_step", "pixel"),
    )
    theta_vals = np.random.uniform(-60, 60, size=(steps, pix))
    phi_vals = np.random.uniform(-60, 60, size=(steps, pix))
    # Ra and Dec pixel shape needs to be the default healpix pixel count
    ra_and_dec = np.random.uniform(-80, 80, size=(steps, pix))
    boundary_scale_factors = np.ones((steps, pix))

    with (
        mock.patch(
            "imap_processing.ultra.l1c.spacecraft_pset.get_spacecraft_pointing_lookup_tables"
        ) as mock_lookup,
    ):
        mock_lookup.return_value = (
            for_indices_by_spin_phase,
            theta_vals,
            phi_vals,
            ra_and_dec,
            boundary_scale_factors,
        )
        yield mock_lookup


@pytest.fixture
def mock_helio_pointing_lookups():
    """Test lookup tables fixture returning an xarray Dataset."""
    np.random.seed(42)
    pix = hp.nside2npix(32)  # reduced for testing
    steps = 2  # Reduced for testing
    energy = 46

    # Ra and Dec pixel shape needs to be the default healpix pixel count
    ra_and_dec = np.random.uniform(-80, 80, size=(steps, pix))

    index_map = np.random.choice([True, False], size=(steps, energy, pix), p=[0.1, 0.9])
    index_map = index_map.astype(bool)
    theta_map = np.random.uniform(-60, 60, size=(steps, energy, pix))
    phi_map = np.random.uniform(-60, 60, size=(steps, energy, pix))
    bsf_map = np.ones((steps, energy, pix))
    with (
        mock.patch(
            "imap_processing.ultra.l1c.helio_pset.make_helio_index_maps_with_nominal_kernels"
        ) as mock_lookup,
    ):
        ds = xr.Dataset(
            data_vars={
                "index": (
                    ["spin_phase_step", "energy", "pixel"],
                    index_map,
                    {"long_name": "Pixel in FOV flag"},
                ),
                "theta": (
                    ["spin_phase_step", "energy", "pixel"],
                    theta_map,
                    {"long_name": "Instrument theta angle", "units": "degrees"},
                ),
                "phi": (
                    ["spin_phase_step", "energy", "pixel"],
                    phi_map,
                    {"long_name": "Instrument phi angle", "units": "degrees"},
                ),
                "bsf": (
                    ["spin_phase_step", "energy", "pixel"],
                    bsf_map,
                    {"long_name": "Boundary scale factor", "units": "fractional"},
                ),
                "ra_and_dec": (["spin_phase_step", "pixel"], ra_and_dec),
            },
            coords={
                "spin_phase_step": np.arange(steps),
                "energy": np.arange(energy),
                "pixel": np.arange(pix),
            },
        )
        mock_lookup.return_value = ds

        yield mock_lookup


@pytest.fixture
def mock_goodtimes_dataset():
    """Create a mock goodtimes dataset."""
    # Set up bit flags
    intervals, _, _ = build_energy_bins()
    energy_ranges = get_binned_energy_ranges(intervals)
    energy_flags = get_energy_range_flags(energy_ranges)
    nspins = 100
    flags = 2 ** np.arange(9)
    quality = np.zeros(nspins, dtype=np.uint16)
    quality[0] = flags[0]  # Set the first flag for the first spin
    quality[1] = flags[1]  # Set the second flag for the second
    quality[2] = flags[2]  # Set the third flag for the third spin
    return xr.Dataset(
        {
            "spin_number": ("epoch", np.zeros(nspins)),
            "energy_range_flags": ("energy_flags", energy_flags),
            "quality_low_voltage": ("spin_number", quality),
            "quality_high_energy": ("spin_number", np.zeros(nspins, dtype=np.uint16)),
            "quality_statistics": ("spin_number", np.zeros(nspins, dtype=np.uint16)),
            "energy_range_edges": ("energy_ranges", energy_ranges),
            "spin_period": (
                "spin_number",
                np.full(nspins, 15),
            ),  # nominal spin period of 15 seconds
        }
    )

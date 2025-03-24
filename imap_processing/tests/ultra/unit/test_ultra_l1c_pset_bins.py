"Tests pointing sets"

import astropy_healpix.healpy as hp
import cdflib
import numpy as np
import pandas as pd
import pytest
from cdflib import CDF

from imap_processing import imap_module_directory
from imap_processing.ena_maps.utils.spatial_utils import build_spatial_bins
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import (
    build_energy_bins,
    get_helio_exposure_times,
    get_spacecraft_exposure_times,
    get_spacecraft_histogram,
    get_spacecraft_sensitivity,
)

BASE_PATH = imap_module_directory / "ultra" / "lookup_tables"
TEST_PATH = imap_module_directory / "tests" / "ultra" / "test_data" / "l1"


@pytest.fixture()
def test_data():
    """Test data fixture."""
    vx_sc = np.array([-186.5575, 508.5697, 508.5697, 508.5697])
    vy_sc = np.array([-707.5707, -516.0282, -516.0282, -516.0282])
    vz_sc = np.array([618.0569, 892.6931, 892.6931, 892.6931])
    energy = np.array([3.384, 3.385, 4.138, 4.138])
    v = np.column_stack((vx_sc, vy_sc, vz_sc))

    return v, energy


def test_build_energy_bins():
    """Tests build_energy_bins function."""
    energy_bin_edges, energy_midpoints = build_energy_bins()
    energy_bin_start = [interval[0] for interval in energy_bin_edges]
    energy_bin_end = [interval[1] for interval in energy_bin_edges]

    assert energy_bin_start[0] == 0
    assert energy_bin_start[1] == 3.385
    assert len(energy_bin_edges) == 24
    assert energy_midpoints[0] == (energy_bin_start[0] + energy_bin_end[0]) / 2

    # Comparison to expected values.
    np.testing.assert_allclose(energy_bin_end[1], 4.137, atol=1e-4)
    np.testing.assert_allclose(energy_bin_start[-1], 279.810, atol=1e-4)
    np.testing.assert_allclose(energy_bin_end[-1], 341.989, atol=1e-4)


def test_get_spacecraft_histogram(test_data):
    """Tests get_histogram function."""
    v, energy = test_data

    energy_bin_edges, _ = build_energy_bins()
    subset_energy_bin_edges = energy_bin_edges[:3]

    hist = get_spacecraft_histogram(v, energy, subset_energy_bin_edges, nside=1)
    assert hist.shape == (hp.nside2npix(1), len(subset_energy_bin_edges))

    # Spot check that 2 counts are in the third energy bin
    assert np.sum(hist[:, 2]) == 2

    # Test overlapping energy bins
    overlapping_bins = [
        (0.0, 3.385),
        (2.5, 4.137),
        (3.385, 5.057),
    ]
    hist = get_spacecraft_histogram(v, energy, overlapping_bins, nside=1)
    # Spot check that 3 counts are in the third energy bin
    assert np.sum(hist[:, 2]) == 3


@pytest.mark.external_test_data()
def test_get_spacecraft_exposure_times():
    """Test get_spacecraft_exposure_times function."""
    constant_exposure = TEST_PATH / "ultra_90_dps_exposure.csv"
    df_exposure = pd.read_csv(constant_exposure)
    exposure_pointing = get_spacecraft_exposure_times(df_exposure)
    assert exposure_pointing.shape == (196608,)

    np.testing.assert_allclose(
        exposure_pointing.values[22684:22686],
        np.array([1.035, 1.035]) * 5760,
        atol=1e-6,
    )


@pytest.mark.external_kernel()
@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
def test_get_helio_exposure_times():
    """Tests get_helio_exposure_times function."""

    constant_exposure = BASE_PATH / "dps_grid45_compressed.cdf"
    start_time = 829485054.185627
    end_time = 829567884.185627
    mid_time = np.average([start_time, end_time])

    with cdflib.CDF(constant_exposure) as cdf_file:
        sc_exposure = cdf_file.varget("dps_grid45")

    exposure_3d = get_helio_exposure_times(mid_time, sc_exposure)

    energy_bin_edges, energy_midpoints = build_energy_bins()
    az_bin_edges, el_bin_edges, az_bin_midpoints, el_bin_midpoints = (
        build_spatial_bins()
    )

    assert exposure_3d.shape == (
        len(el_bin_midpoints),
        len(az_bin_midpoints),
        len(energy_midpoints),
    )

    cdf_files = [
        ("dps_exposure_helio_45_E1.cdf", "dps_exposure_helio_45_E1"),
        ("dps_exposure_helio_45_E12.cdf", "dps_exposure_helio_45_E12"),
        ("dps_exposure_helio_45_E24.cdf", "dps_exposure_helio_45_E24"),
    ]

    cdf_directory = imap_module_directory / "tests" / "ultra" / "test_data" / "l1"

    exposures = []

    for file_name, var_name in cdf_files:
        file_path = cdf_directory / file_name
        with CDF(file_path) as cdf_file:
            exposure_data = cdf_file.varget(var_name)
            transposed_exposure = np.transpose(exposure_data, (2, 1, 0))
            exposures.append(transposed_exposure)

    assert np.array_equal(np.squeeze(exposures[0]), exposure_3d[:, :, 0])
    assert np.array_equal(np.squeeze(exposures[1]), exposure_3d[:, :, 11])
    assert np.array_equal(np.squeeze(exposures[2]), exposure_3d[:, :, 23])


@pytest.mark.external_test_data()
def test_get_spacecraft_sensitivity():
    """Tests get_spacecraft_sensitivity function."""
    # TODO: remove below here with lookup table aux api
    efficiences = TEST_PATH / "Ultra_90_DPS_efficiencies_all.csv"
    geometric_function = TEST_PATH / "ultra_90_dps_gf.csv"

    df_efficiencies = pd.read_csv(efficiences)
    df_geometric_function = pd.read_csv(geometric_function)

    sensitivity = get_spacecraft_sensitivity(df_efficiencies, df_geometric_function)

    assert sensitivity.shape == df_efficiencies.shape

    df_efficiencies_test = pd.DataFrame(
        {"3.0keV": [1.0, 2.0], "3.5keV": [3.0, 4.0], "4.0keV": [5.0, 6.0]}
    )

    df_geometric_function_test = pd.DataFrame({"Response": [0.1, 0.2]})

    df_sensitivity_test = df_efficiencies_test.mul(
        df_geometric_function_test["Response"], axis=0
    )

    expected_sensitivity = pd.DataFrame(
        {"3.0keV": [0.1, 0.4], "3.5keV": [0.3, 0.8], "4.0keV": [0.5, 1.2]}
    )

    assert np.allclose(
        df_sensitivity_test.to_numpy(), expected_sensitivity.to_numpy(), atol=1e-6
    )

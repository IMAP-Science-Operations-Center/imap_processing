"Tests pointing sets"

import astropy_healpix.healpy as hp
import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import (
    build_energy_bins,
    get_background_rates,
    get_helio_exposure_times,
    get_spacecraft_exposure_times,
    get_spacecraft_histogram,
    get_spacecraft_sensitivity,
    grid_sensitivity,
)

BASE_PATH = imap_module_directory / "ultra" / "lookup_tables"
TEST_PATH = imap_module_directory / "tests" / "ultra" / "data" / "l1"


@pytest.fixture
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
    intervals, energy_midpoints, energy_bin_geometric_means = build_energy_bins()
    energy_bin_start = [interval[0] for interval in intervals]
    energy_bin_end = [interval[1] for interval in intervals]

    assert energy_bin_start[0] == 0
    assert energy_bin_start[1] == 3.385
    assert len(intervals) == 24
    assert energy_midpoints[0] == (energy_bin_start[0] + energy_bin_end[0]) / 2

    # Comparison to expected values.
    np.testing.assert_allclose(energy_bin_end[1], 4.137, atol=1e-4)
    np.testing.assert_allclose(energy_bin_start[-1], 279.810, atol=1e-4)
    np.testing.assert_allclose(energy_bin_end[-1], 341.989, atol=1e-4)

    expected_geometric_means = np.sqrt(
        np.array(energy_bin_start) * np.array(energy_bin_end)
    )
    np.testing.assert_allclose(
        energy_bin_geometric_means, expected_geometric_means, atol=1e-4
    )


def test_get_spacecraft_histogram(test_data):
    """Tests get_histogram function."""
    v, energy = test_data

    energy_bin_edges, _, _ = build_energy_bins()
    subset_energy_bin_edges = energy_bin_edges[:3]

    hist, latitude, longitude, n_pix = get_spacecraft_histogram(
        v, energy, subset_energy_bin_edges, nside=1
    )
    assert hist.shape == (len(subset_energy_bin_edges), hp.nside2npix(1))
    assert n_pix == hp.nside2npix(1)
    assert latitude.shape == (n_pix,)
    assert longitude.shape == (n_pix,)

    # Spot check that 2 counts are in the third energy bin
    assert np.sum(hist[2, :]) == 2

    # Test overlapping energy bins
    overlapping_bins = [
        (0.0, 3.385),
        (2.5, 4.137),
        (3.385, 5.057),
    ]
    hist, latitude, longitude, n_pix = get_spacecraft_histogram(
        v, energy, overlapping_bins, nside=1
    )
    # Spot check that 3 counts are in the third energy bin
    assert np.sum(hist[2, :]) == 3
    assert n_pix == hp.nside2npix(1)
    assert latitude.shape == (n_pix,)
    assert longitude.shape == (n_pix,)


def test_get_background_rates():
    """Tests get_background_rates function."""
    background_rates = get_background_rates(nside=128)
    assert background_rates.shape == hp.nside2npix(128)


@pytest.mark.external_test_data
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


@pytest.mark.external_kernel
@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
def test_get_helio_exposure_times():
    """Tests get_helio_exposure_times function."""

    start_time = 829485054.185627
    end_time = 829567884.185627

    mid_time = np.average([start_time, end_time])

    constant_exposure = TEST_PATH / "ultra_90_dps_exposure.csv"
    df_exposure = pd.read_csv(constant_exposure)

    helio_exposure = get_helio_exposure_times(mid_time, df_exposure)

    _, energy_midpoints, _ = build_energy_bins()

    nside = 128
    npix = hp.nside2npix(nside)
    assert helio_exposure.shape == (npix, len(energy_midpoints))

    total_input = np.sum(df_exposure["Exposure Time"].values)
    total_output = np.sum(helio_exposure[:, 23])

    assert np.allclose(total_input, total_output, atol=1e-6)


@pytest.mark.external_test_data
def test_get_spacecraft_sensitivity():
    """Tests get_spacecraft_sensitivity function."""
    # TODO: remove below here with lookup table aux api
    efficiencies = TEST_PATH / "Ultra_90_DPS_efficiencies_all.csv"
    geometric_function = TEST_PATH / "ultra_90_dps_gf.csv"

    df_efficiencies = pd.read_csv(efficiencies)
    df_geometric_function = pd.read_csv(geometric_function)

    sensitivity, energy_vals, right_ascension, declination = get_spacecraft_sensitivity(
        df_efficiencies, df_geometric_function
    )

    assert sensitivity.shape == (df_efficiencies.shape[0], df_efficiencies.shape[1] - 2)
    assert np.array_equal(energy_vals, np.arange(3.0, 80.5, 0.5))

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

    expected_result = sensitivity["3.0keV"].values
    result = grid_sensitivity(df_efficiencies, df_geometric_function, 3.0)

    assert np.allclose(result, expected_result, atol=1e-5)

    # Check that out-of-bounds energy returns all NaNs
    result = grid_sensitivity(df_efficiencies, df_geometric_function, 2.5)
    assert np.isnan(result).all()

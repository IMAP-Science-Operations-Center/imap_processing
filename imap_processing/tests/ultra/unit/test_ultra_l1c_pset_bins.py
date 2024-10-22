"Tests bins for pointing sets"

import cdflib
import numpy as np
import pytest
import spiceypy as spice
from cdflib import CDF

from imap_processing import imap_module_directory
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import (
    build_energy_bins,
    build_spatial_bins,
    cartesian_to_spherical,
    get_helio_exposure_times,
    get_histogram,
    get_pointing_frame_exposure_times,
    get_pointing_frame_sensitivity,
)

BASE_PATH = imap_module_directory / "ultra" / "lookup_tables"


@pytest.fixture()
def test_data():
    """Test data fixture."""
    vx_sc = np.array([-186.5575, 508.5697, 508.5697, 508.5697])
    vy_sc = np.array([-707.5707, -516.0282, -516.0282, -516.0282])
    vz_sc = np.array([618.0569, 892.6931, 892.6931, 892.6931])
    energy = np.array([3.384, 3.385, 200, 200])
    v = np.column_stack((vx_sc, vy_sc, vz_sc))

    return v, energy


@pytest.fixture()
def kernels(spice_test_data_path):
    """List SPICE kernels."""
    required_kernels = [
        "imap_science_0001.tf",
        "imap_sclk_0000.tsc",
        "sim_1yr_imap_attitude.bc",
        "imap_wkcp.tf",
        "naif0012.tls",
        "sim_1yr_imap_pointing_frame.bc",
        "de440s.bsp",
        "imap_spk_demo.bsp",
    ]
    kernels = [str(spice_test_data_path / kernel) for kernel in required_kernels]

    return kernels


def test_build_energy_bins():
    """Tests build_energy_bins function."""
    energy_bin_edges = build_energy_bins()
    energy_bin_start = energy_bin_edges[:-1]
    energy_bin_end = energy_bin_edges[1:]

    assert energy_bin_start[0] == 0
    assert energy_bin_start[1] == 3.385
    assert len(energy_bin_edges) == 25

    # Comparison to expected values.
    np.testing.assert_allclose(energy_bin_end[1], 4.137, atol=1e-4)
    np.testing.assert_allclose(energy_bin_start[-1], 279.810, atol=1e-4)
    np.testing.assert_allclose(energy_bin_end[-1], 341.989, atol=1e-4)


def test_build_spatial_bins():
    """Tests build_spatial_bins function."""
    az_bin_edges, el_bin_edges, az_bin_midpoints, el_bin_midpoints = (
        build_spatial_bins()
    )

    assert az_bin_edges[0] == 0
    assert az_bin_edges[-1] == 360
    assert len(az_bin_edges) == 721

    assert el_bin_edges[0] == -90
    assert el_bin_edges[-1] == 90
    assert len(el_bin_edges) == 361

    assert len(az_bin_midpoints) == 720
    np.testing.assert_allclose(az_bin_midpoints[0], 0.25, atol=1e-4)
    np.testing.assert_allclose(az_bin_midpoints[-1], 359.75, atol=1e-4)

    assert len(el_bin_midpoints) == 360
    np.testing.assert_allclose(el_bin_midpoints[0], -89.75, atol=1e-4)
    np.testing.assert_allclose(el_bin_midpoints[-1], 89.75, atol=1e-4)


def test_cartesian_to_spherical(test_data):
    """Tests cartesian_to_spherical function."""
    v, _ = test_data

    az_sc, el_sc, r = cartesian_to_spherical(v)

    # MATLAB code outputs:
    np.testing.assert_allclose(
        np.unique(np.radians(az_sc)), np.array([1.31300, 2.34891]), atol=1e-05, rtol=0
    )
    np.testing.assert_allclose(
        np.unique(np.radians(el_sc)), np.array([-0.88901, -0.70136]), atol=1e-05, rtol=0
    )


def test_get_histogram(test_data):
    """Tests get_histogram function."""
    v, energy = test_data

    az_bin_edges, el_bin_edges, az_bin_midpoints, el_bin_midpoints = (
        build_spatial_bins()
    )
    energy_bin_edges = build_energy_bins()

    hist = get_histogram(v, energy, az_bin_edges, el_bin_edges, energy_bin_edges)

    assert hist.shape == (
        len(az_bin_edges) - 1,
        len(el_bin_edges) - 1,
        len(energy_bin_edges) - 1,
    )


def test_get_pointing_frame_exposure_times():
    """Tests get_pointing_frame_exposure_times function."""

    constant_exposure = BASE_PATH / "dps_grid45_compressed.cdf"
    spins_per_pointing = 5760
    exposure = get_pointing_frame_exposure_times(
        constant_exposure, spins_per_pointing, "45"
    )

    assert exposure.shape == (720, 360)
    # Assert that the exposure time at the highest azimuth is
    # 15s x spins per pointing.
    assert np.array_equal(
        exposure[:, 359], np.full_like(exposure[:, 359], spins_per_pointing * 15)
    )
    # Assert that the exposure time at the lowest azimuth is 0 (no exposure).
    assert np.array_equal(exposure[:, 0], np.full_like(exposure[:, 359], 0.0))


@pytest.mark.external_kernel()
def test_et_helio_exposure_times(kernels):
    """Tests get_helio_exposure_times function."""

    spice.furnsh(kernels)
    constant_exposure = BASE_PATH / "dps_grid45_compressed.cdf"
    start_time = 829485054.185627
    end_time = 829567884.185627
    mid_time = np.average([start_time, end_time])

    with cdflib.CDF(constant_exposure) as cdf_file:
        sc_exposure = cdf_file.varget(f"dps_grid45")

    # Call the function to get the computed 3D exposure array
    exposure_3d = get_helio_exposure_times(mid_time, sc_exposure)

    # Rebuild the bin edges and midpoints for validation
    energy_bin_edges, energy_midpoints = build_energy_bins()
    az_bin_edges, el_bin_edges, az_bin_midpoints, el_bin_midpoints = (
        build_spatial_bins()
    )

    assert exposure_3d.shape == (len(el_bin_midpoints), len(az_bin_midpoints), len(energy_midpoints))

    cdf_files = [
        ("dps_exposure_helio_45_E1.cdf", "dps_exposure_helio_45_E1"),
        ("dps_exposure_helio_45_E12.cdf", "dps_exposure_helio_45_E12"),
        ("dps_exposure_helio_45_E24.cdf", "dps_exposure_helio_45_E24")
    ]

    cdf_directory = imap_module_directory / "tests" / "ultra" / "test_data" / "l1"

    exposures = []

    for file_name, var_name in cdf_files:
        file_path = cdf_directory / file_name
        with CDF(file_path) as cdf_file:
            exposure_data = cdf_file.varget(var_name)
            transposed_exposure = np.transpose(exposure_data, (2, 1, 0))
            exposures.append(transposed_exposure)

    np.array_equal(exposures[0], exposure_3d[:, :, 0])
    np.array_equal(exposures[1], exposure_3d[:, :, 1])
    np.array_equal(exposures[2], exposure_3d[:, :, 2])


def test_get_pointing_frame_sensitivity():
    """Tests get_pointing_frame_sensitivity function."""

    # TODO: energy bins need to be modified from N=90 to N=24.
    constant_sensitivity = BASE_PATH / "dps_sensitivity45.cdf"
    spins_per_pointing = 5760
    sensitivity = get_pointing_frame_sensitivity(
        constant_sensitivity, spins_per_pointing, "45",
    )

    assert sensitivity.shape == (90, 720, 360)

    with cdflib.CDF(constant_sensitivity) as cdf_file:
        expected_sensitivity = cdf_file.varget(f"dps_sensitivity45") * spins_per_pointing

    assert np.array_equal(sensitivity, expected_sensitivity)

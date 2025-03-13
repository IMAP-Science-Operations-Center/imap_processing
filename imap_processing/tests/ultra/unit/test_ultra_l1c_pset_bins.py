"Tests pointing sets"

from pathlib import Path

import astropy_healpix.healpy as hp
import cdflib
import numpy as np
import pytest
from cdflib import CDF

from imap_processing import imap_module_directory
from imap_processing.ena_maps.utils.spatial_utils import build_spatial_bins
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import (
    build_energy_bins,
    get_helio_exposure_times,
    get_pointing_frame_sensitivity,
    get_spacecraft_exposure_times,
    get_spacecraft_histogram,
)

BASE_PATH = imap_module_directory / "ultra" / "lookup_tables"


@pytest.fixture()
def test_data():
    """Test data fixture."""
    vx_sc = np.array([-186.5575, 508.5697, 508.5697, 508.5697])
    vy_sc = np.array([-707.5707, -516.0282, -516.0282, -516.0282])
    vz_sc = np.array([618.0569, 892.6931, 892.6931, 892.6931])
    energy = np.array([3.384, 3.385, 4.138, 4.138])
    v = np.column_stack((vx_sc, vy_sc, vz_sc))

    return v, energy


@pytest.fixture()
def fake_cdf_exposure_data(tmpdir):
    """Test exposure data fixture."""
    exposure_time = np.array([0, 2, 4, 1, 1, 3, 6, 0, 0, 0, 0, 0])

    cdf_path = Path(tmpdir) / "fake_exposure.cdf"

    var_specs = [
        {
            "Variable": "exposure_time",
            "Data_Type": 21,
            "Num_Elements": 1,
            "Rec_Vary": True,
            "Dim_Sizes": [],
        },
    ]

    cdf = cdflib.cdfwrite.CDF(str(cdf_path))

    for var_spec, var_data in zip(var_specs, [exposure_time]):
        cdf.write_var(var_spec, var_data=var_data)

    cdf.close()

    return cdf_path, exposure_time


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


def test_get_spacecraft_exposure_times(fake_cdf_exposure_data):
    """Test get_spacecraft_exposure_times function."""
    constant_exposure = BASE_PATH / "ultra_90_dps_exposure_compressed.cdf"
    exposure_pointing = get_spacecraft_exposure_times(constant_exposure)
    assert exposure_pointing.shape == (196608,)

    cdf_path, expected_exposure_time = fake_cdf_exposure_data

    exposure_pointing = get_spacecraft_exposure_times(cdf_path)

    np.testing.assert_allclose(
        exposure_pointing, expected_exposure_time * 5760, atol=1e-6
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


def test_get_pointing_frame_sensitivity():
    """Tests get_pointing_frame_sensitivity function."""

    # TODO: energy bins need to be modified from N=90 to N=24.
    constant_sensitivity = BASE_PATH / "dps_sensitivity45.cdf"
    spins_per_pointing = 5760
    sensitivity = get_pointing_frame_sensitivity(
        constant_sensitivity,
        spins_per_pointing,
        "45",
    )

    assert sensitivity.shape == (90, 720, 360)

    with cdflib.CDF(constant_sensitivity) as cdf_file:
        expected_sensitivity = cdf_file.varget("dps_sensitivity45") * spins_per_pointing

    assert np.array_equal(sensitivity, expected_sensitivity)

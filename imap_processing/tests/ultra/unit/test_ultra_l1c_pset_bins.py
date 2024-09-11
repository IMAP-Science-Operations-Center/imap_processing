"Tests bins for pointing sets"

import numpy as np

from imap_processing.ultra.l1c.ultra_l1c_pset_bins import (
    bin_space,
    build_energy_bins,
    build_spatial_bins,
    cartesian_to_spherical,
)


def test_build_energy_bins():
    """Tests build_energy_bins function."""
    energy_bin_start, energy_bin_end = build_energy_bins()

    assert energy_bin_start[0] == 3.5
    assert len(energy_bin_start) == 90
    assert len(energy_bin_end) == 90

    # Comparison to expected values.
    np.testing.assert_allclose(energy_bin_end[0], 3.6795, atol=1e-4)
    np.testing.assert_allclose(energy_bin_start[-1], 299.9724, atol=1e-4)
    np.testing.assert_allclose(energy_bin_end[-1], 315.3556, atol=1e-4)


def test_build_spatial_bins():
    """Tests build_spatial_bins function."""
    az_bin_edges, el_bin_edges = build_spatial_bins()

    assert az_bin_edges[0] == 0
    assert az_bin_edges[-1] == 360
    assert len(az_bin_edges) == 721

    assert el_bin_edges[0] == -90
    assert el_bin_edges[-1] == 90
    assert len(el_bin_edges) == 361


def test_cartesian_to_spherical():
    """Tests cartesian_to_spherical function."""
    # Example particle velocity in the pointing frame wrt s/c.
    vx_sc = np.array([-186.5575, 508.5697])
    vy_sc = np.array([-707.5707, -516.0282])
    vz_sc = np.array([618.0569, 892.6931])

    az, el = cartesian_to_spherical(vx_sc, vy_sc, vz_sc)

    # MATLAB code outputs:
    assert az == np.array([1.313003856057083, 2.348915185230239])
    assert el == np.array([-0.701366480008680, -0.889015692861197])


def test_bin_space():
    """Tests bin_space function."""
    # Example particle velocity in the pointing frame wrt s/c.
    vx_sc = np.array([-186.5575, 508.5697])
    vy_sc = np.array([-707.5707, -516.0282])
    vz_sc = np.array([618.0569, 892.6931])

    az_midpoint, el_midpoint = bin_space(vx_sc, vy_sc, vz_sc)
    az, el = cartesian_to_spherical(vx_sc, vy_sc, vz_sc)

    az_within_tolerance = np.abs(az - az_midpoint) <= 0.25
    el_within_tolerance = np.abs(el - el_midpoint) <= 0.25

    assert np.all(az_within_tolerance)
    assert np.all(el_within_tolerance)

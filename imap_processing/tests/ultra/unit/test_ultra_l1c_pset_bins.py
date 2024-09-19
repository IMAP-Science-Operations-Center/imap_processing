"Tests bins for pointing sets"

import numpy as np

from imap_processing.ultra.l1c.ultra_l1c_pset_bins import (
    bin_energy,
    bin_space,
    build_energy_bins,
    build_spatial_bins,
    cartesian_to_spherical,
)


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


def test_cartesian_to_spherical():
    """Tests cartesian_to_spherical function."""
    # Example particle velocity in the pointing frame wrt s/c.
    vx_sc = np.array([-186.5575, 508.5697])
    vy_sc = np.array([-707.5707, -516.0282])
    vz_sc = np.array([618.0569, 892.6931])
    v = np.column_stack((vx_sc, vy_sc, vz_sc))

    az_sc, el_sc, r = cartesian_to_spherical(v)

    # MATLAB code outputs:
    np.testing.assert_allclose(az_sc, np.array([1.31300, 2.34891]), atol=1e-05, rtol=0)
    np.testing.assert_allclose(
        el_sc, np.array([-0.70136, -0.88901]), atol=1e-05, rtol=0
    )


def test_bin_space():
    """Tests bin_space function."""
    # Example particle velocity in the pointing frame wrt s/c.
    vx_sc = np.array([-186.5575, 508.5697, 508.5697, 0])
    vy_sc = np.array([-707.5707, -516.0282, -516.0282, 0])
    vz_sc = np.array([618.0569, 892.6931, 892.6931, -1])

    v = np.column_stack((vx_sc, vy_sc, vz_sc))
    # 259200
    bin_id = bin_space(v)

    az_bin_edges, el_bin_edges, az_bin_midpoints, el_bin_midpoints = (
        build_spatial_bins()
    )
    expected_az, expected_el, _ = cartesian_to_spherical(v)

    expected_az_degrees = np.degrees(expected_az)
    expected_el_degrees = np.degrees(expected_el)

    # Assert that we can back-calculate the bin.
    az_indices = bin_id // len(el_bin_midpoints)
    el_indices = bin_id % len(el_bin_midpoints)

    # Make certain this is binned properly.
    az_bin = az_bin_midpoints[az_indices]
    el_bin = el_bin_midpoints[el_indices]

    print("hi")


def test_bin_energy():
    """Tests bin_energy function."""
    energy = np.array([3.384, 3.385, 341.989, 342])
    bin = bin_energy(energy)

    np.testing.assert_equal(bin, (0, 3.385, 341.989, 341.989))

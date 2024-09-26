"Tests bins for pointing sets"

import numpy as np
import pytest

from imap_processing.ultra.l1c.ultra_l1c_pset_bins import (
    bin_data,
    build_energy_bins,
    build_spatial_bins,
    cartesian_to_spherical,
    create_unique_identifiers,
    extract_non_zero_indices_and_counts,
    get_histogram,
)


@pytest.fixture()
def test_data():
    """Test data fixture."""
    vx_sc = np.array([-186.5575, 508.5697, 508.5697, 508.5697])
    vy_sc = np.array([-707.5707, -516.0282, -516.0282, -516.0282])
    vz_sc = np.array([618.0569, 892.6931, 892.6931, 892.6931])
    energy = np.array([3.384, 3.385, 200, 200])
    v = np.column_stack((vx_sc, vy_sc, vz_sc))

    return v, energy


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
    _, non_zero_counts = extract_non_zero_indices_and_counts(hist)

    assert hist.shape == (
        len(az_bin_edges) - 1,
        len(el_bin_edges) - 1,
        len(energy_bin_edges) - 1,
    )
    assert hist.sum() == np.sum(non_zero_counts)

    az_indices, el_indices, energy_bin_id = np.argwhere(hist > 0).T
    az, el, _ = cartesian_to_spherical(v)
    assert np.allclose(
        np.unique(az), np.unique(az_bin_midpoints[az_indices]), atol=0.25
    )
    assert np.allclose(
        np.unique(el), np.unique(el_bin_midpoints[el_indices]), atol=0.25
    )

    for az_idx, el_idx, energy_idx, count in zip(
        az_indices, el_indices, energy_bin_id, non_zero_counts
    ):
        assert hist[az_idx, el_idx, energy_idx] == count


def test_create_unique_identifiers(test_data):
    """Tests create_unique_identifiers function."""
    v, energy = test_data

    az_bin_edges, el_bin_edges, az_bin_midpoints, el_bin_midpoints = (
        build_spatial_bins()
    )
    energy_bin_edges = build_energy_bins()

    hist = get_histogram(v, energy, az_bin_edges, el_bin_edges, energy_bin_edges)
    _, non_zero_counts = extract_non_zero_indices_and_counts(hist)

    counts, bin_id, midpoints = create_unique_identifiers(
        hist, az_bin_midpoints, el_bin_midpoints
    )

    assert counts.size == len(bin_id[0]) * len(bin_id[1]) * len(bin_id[2])
    assert sum(non_zero_counts) == np.sum(counts)
    assert np.array_equal(midpoints[0], az_bin_midpoints[np.unique(bin_id[0])])
    assert np.array_equal(midpoints[1], el_bin_midpoints[np.unique(bin_id[1])])


def test_bin_data(test_data):
    """Tests bin_data function."""
    v, energy = test_data

    counts, bin_id, midpoints, energy_edge_start = bin_data(v, energy)
    az_bin_edges, el_bin_edges, az_bin_midpoints, el_bin_midpoints = (
        build_spatial_bins()
    )
    energy_bin_edges = build_energy_bins()

    hist = get_histogram(v, energy, az_bin_edges, el_bin_edges, energy_bin_edges)

    az_indices = bin_id[0]  # Attitude indices
    el_indices = bin_id[1]  # Elevation indices
    energy_indices = bin_id[2]  # Energy bin indices

    for i, az_idx in enumerate(np.unique(az_indices)):
        for j, el_idx in enumerate(np.unique(el_indices)):
            for k, energy_idx in enumerate(np.unique(energy_indices)):
                expected_count = hist[az_idx, el_idx, energy_idx]
                actual_count = counts[i, j, k]
                assert actual_count == expected_count


def test_extract_non_zero_indices_and_counts():
    """Tests extract_non_zero_indices_and_counts function."""
    hist = np.array(
        [
            [[0, 0, 0], [1, 0, 2]],
            [[0, 3, 0], [0, 0, 0]],
        ]
    )

    expected_indices = np.array(
        [
            [0, 0, 1],
            [1, 1, 0],
            [0, 2, 1],
        ]
    )

    expected_counts = np.array([1, 2, 3])

    non_zero_indices, non_zero_counts = extract_non_zero_indices_and_counts(hist)
    np.testing.assert_array_equal(non_zero_indices, expected_indices)
    np.testing.assert_array_equal(non_zero_counts, expected_counts)

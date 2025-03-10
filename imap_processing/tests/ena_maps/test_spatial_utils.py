"""Test creation of solid angle map and other spatial utils."""

import numpy as np
import numpy.testing as npt
import pytest

from imap_processing.ena_maps.utils import spatial_utils

# Parameterize with spacings (degrees here):
valid_spacings = [0.25, 0.5, 1, 5, 10, 20]
invalid_spacings = [0, -1, 11]
invalid_spacings_match_str = [
    "Spacing must be positive valued, non-zero.",
    "Spacing must be positive valued, non-zero.",
    "Spacing must divide evenly into pi radians.",
]


def test_build_spatial_bins():
    """Tests build_spatial_bins function."""
    az_bin_edges, el_bin_edges, az_bin_midpoints, el_bin_midpoints = (
        spatial_utils.build_spatial_bins(
            az_spacing_deg=0.5,
            el_spacing_deg=0.5,
        )
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


@pytest.mark.parametrize("spacing", valid_spacings)
def test_build_solid_angle_map_integration(spacing):
    """Test build_solid_angle_map function integrates to 4 pi steradians."""
    solid_angle_map_steradians = spatial_utils.build_solid_angle_map(
        spacing_deg=spacing
    )
    np.testing.assert_allclose(
        np.sum(solid_angle_map_steradians), 4 * np.pi, atol=0, rtol=1e-9
    )


@pytest.mark.parametrize("spacing", valid_spacings)
def test_build_solid_angle_map_equal_at_equal_el(spacing):
    """Test build_solid_angle_map function produces equal solid angle at equal el."""
    solid_angle_map = spatial_utils.build_solid_angle_map(
        spacing_deg=spacing,
    )
    el_grid = spatial_utils.AzElSkyGrid(
        spacing_deg=spacing,
        reversed_elevation=False,
    ).el_grid
    for unique_el in np.unique(el_grid):
        solid_angles = solid_angle_map[el_grid == unique_el]
        np.testing.assert_allclose(solid_angles, solid_angles[0])


@pytest.mark.parametrize(
    "spacing, match_str", zip(invalid_spacings, invalid_spacings_match_str)
)
def test_build_solid_angle_map_invalid_spacing(spacing, match_str):
    """Test build_solid_angle_map function raises error for invalid spacing."""
    with pytest.raises(ValueError, match=match_str):
        _ = spatial_utils.build_solid_angle_map(
            spacing_deg=spacing,
        )


@pytest.mark.parametrize("order", ["C", "F"])
def test_rewrap_even_spaced_az_el_grid_1d(order):
    """Test rewrap_even_spaced_az_el_grid function, without extra axis."""
    orig_shape = (360 * 12, 180 * 12)
    orig_grid = np.fromfunction(lambda i, j: i**2 + j, orig_shape, dtype=int)
    raveled_values = orig_grid.ravel(order=order)
    rewrapped_grid_infer_shape = spatial_utils.rewrap_even_spaced_az_el_grid(
        raveled_values,
        order=order,
    )
    rewrapped_grid_known_shape = spatial_utils.rewrap_even_spaced_az_el_grid(
        raveled_values,
        grid_shape=orig_shape,
        order=order,
    )

    assert np.array_equal(rewrapped_grid_infer_shape, orig_grid)
    assert np.array_equal(rewrapped_grid_known_shape, orig_grid)


@pytest.mark.parametrize("order", ["C", "F"])
def test_rewrap_even_spaced_az_el_grid_2d(order):
    """Test rewrap_even_spaced_az_el_grid function, with extra axis."""
    orig_shape = (5, 360 * 12, 180 * 12)
    orig_grid = np.fromfunction(lambda i, j, k: i**2 + j + k, orig_shape, dtype=int)
    raveled_values = orig_grid.reshape(5, -1, order=order)
    rewrapped_grid_infer_shape = spatial_utils.rewrap_even_spaced_az_el_grid(
        raveled_values,
        order=order,
    )
    rewrapped_grid_known_shape = spatial_utils.rewrap_even_spaced_az_el_grid(
        raveled_values,
        grid_shape=orig_shape[-2:],
        order=order,
    )
    assert raveled_values.shape == (5, 360 * 12 * 180 * 12)
    assert np.array_equal(rewrapped_grid_infer_shape, orig_grid)
    assert np.array_equal(rewrapped_grid_known_shape, orig_grid)


class TestAzElSkyGrid:
    @pytest.mark.parametrize("spacing", valid_spacings)
    def test_instantiate_and_values(self, spacing):
        grid = spatial_utils.AzElSkyGrid(
            spacing_deg=spacing,
            reversed_elevation=False,
        )

        # Size checks
        assert grid.az_bin_midpoints.size == int(360 / spacing) == grid.grid_shape[0]
        assert grid.el_bin_midpoints.size == int(180 / spacing) == grid.grid_shape[1]
        assert grid.az_grid.shape == grid.el_grid.shape == grid.grid_shape

        # Check grid values in radians
        expected_azimuth_bin_midpoints_deg = np.arange(
            (spacing / 2), 360 + (spacing / 2), spacing
        )
        expected_elevation_bin_midpoints_deg = np.arange(
            -90 + (spacing / 2), 90 + (spacing / 2), spacing
        )
        npt.assert_allclose(
            grid.az_bin_midpoints,
            expected_azimuth_bin_midpoints_deg,
            atol=1e-11,
        )
        npt.assert_allclose(
            grid.el_bin_midpoints,
            expected_elevation_bin_midpoints_deg,
            atol=1e-11,
        )

        # Check bin edges in degrees, radians
        expected_az_bin_edges_deg = np.arange(0, 360 + spacing, spacing)
        expected_el_bin_edges_deg = np.arange(-90, 90 + spacing, spacing)
        npt.assert_allclose(grid.az_bin_edges, expected_az_bin_edges_deg, atol=1e-11)
        npt.assert_allclose(grid.el_bin_edges, expected_el_bin_edges_deg, atol=1e-11)

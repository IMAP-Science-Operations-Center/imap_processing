"Tests bins for pointing sets"

import numpy as np

from imap_processing.ultra.l1c.ultra_l1c_pset_bins import build_energy_bins, build_spatial_bins


def test_build_energy_bins():
    """Tests build_energy_bins function."""
    energy_bin_start, energy_bin_end = build_energy_bins()

    assert energy_bin_start[0] == 3.5
    assert len(energy_bin_start) == 90
    assert len(energy_bin_end) == 90

    # Comparison to expected values
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

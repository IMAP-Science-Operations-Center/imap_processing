import numpy as np
import pytest
import xarray as xr

from imap_processing.lo.l2.lo_l2 import (
    calculate_fluxes,
    calculate_rates,
    lo_l2,
    project_pset_to_rect_map,
)
from imap_processing.spice import geometry


@pytest.fixture
def pset():
    h_counts = np.zeros((1, 3600, 40, 7))
    h_counts[:, :, 0:10, :] = 1

    exposure_time = np.full((1, 3600, 40, 7), 0.5)

    dataset = xr.Dataset(
        {
            "h_counts": (("epoch", "longitude", "latitude", "energy"), h_counts),
            "exposure_time": (
                ("epoch", "longitude", "latitude", "energy"),
                exposure_time,
            ),
        },
        coords={
            "epoch": [8.1794907049e17],
            "longitude": [i for i in range(3600)],
            "latitude": [i for i in range(40)],
            "energy": [i for i in range(1, 8)],
        },
    )
    return dataset


@pytest.mark.external_kernel
@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
def test_project_pset_to_rect_map(pset):
    # Act
    lo_rect_map = project_pset_to_rect_map([pset], 6, geometry.SpiceFrame.ECLIPJ2000)
    assert lo_rect_map.spacing_deg == 6
    assert lo_rect_map.spice_reference_frame == geometry.SpiceFrame.ECLIPJ2000
    assert lo_rect_map.num_points == 1800


@pytest.mark.external_kernel
@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
def test_calculate_rates():
    # Arrange
    counts = np.zeros((1, 7, 1800))
    counts[0, 0, 0] = 1
    counts[0, 0, 1] = 2

    exposure_time = np.full((1, 7, 1800), 0.5)

    expected_rates = np.zeros((1, 7, 1800))
    expected_rates[0, 0, 0] = 2
    expected_rates[0, 0, 1] = 4

    # Act
    h_rate = calculate_rates(counts, exposure_time)

    # Assert
    np.testing.assert_array_equal(h_rate, expected_rates)


@pytest.mark.external_kernel
@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
def test_calculate_fluxes():
    # Arrange
    rates = np.zeros((1, 7, 1800))
    rates[0, 0, 0] = 2
    rates[0, 1, 0] = 12

    expected_fluxes = np.zeros((1, 7, 1800))
    expected_fluxes[0, 0, 0] = 2
    expected_fluxes[0, 1, 0] = 6

    # Act
    flux = calculate_fluxes(rates)

    # Assert
    np.testing.assert_array_equal(flux, expected_fluxes)


@pytest.mark.external_kernel
@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
def test_lo_l2(pset):
    # Arrange
    pset = {"imap_lo_l1c_pset": [pset]}

    # Act
    hflux_map = lo_l2(pset, [])

    # Assert
    assert len(hflux_map) == 1
    assert (
        hflux_map[0].attrs["Logical_source"]
        == "imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-1yr"
    )
    data_vars = ["h_counts", "exposure_time", "h_rate", "h_flux", "solid_angle"]
    for var in data_vars:
        assert var in hflux_map[0].data_vars, f"Variable {var} not found in dataset"
    assert hflux_map[0].data_vars["h_rate"].shape == (1, 7, 60, 30)

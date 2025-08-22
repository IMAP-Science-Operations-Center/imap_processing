import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ena_maps.ena_maps import HealpixSkyMap, RectangularSkyMap
from imap_processing.lo.l1c.lo_l1c import (
    ESA_ENERGY_STEPS,
    N_OFF_ANGLE_BINS,
    N_SPIN_ANGLE_BINS,
    OFF_ANGLE_BIN_CENTERS,
    PSET_DIMS,
    PSET_SHAPE,
    SPIN_ANGLE_BIN_CENTERS,
)
from imap_processing.lo.l2.lo_l2 import (
    add_attributes,
    calculate_fluxes,
    calculate_rates,
    lo_l2,
    project_pset_to_sky_map,
)
from imap_processing.spice import geometry


@pytest.fixture
def pset():
    h_counts = np.zeros(PSET_SHAPE)
    h_counts[:, :, :, 0:10] = 1

    exposure_time = np.full(PSET_SHAPE, 0.5)

    lons, lats = np.meshgrid(
        SPIN_ANGLE_BIN_CENTERS, OFF_ANGLE_BIN_CENTERS, indexing="ij"
    )
    hae_longitude = np.empty((1, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS))
    hae_latitude = np.empty((1, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS))
    hae_longitude[0, :, :] = lons
    hae_latitude[0, :, :] = lats

    dataset = xr.Dataset(
        {
            "h_counts": (
                PSET_DIMS,
                h_counts,
            ),
            "exposure_time": (
                PSET_DIMS,
                exposure_time,
            ),
            "hae_longitude": (("epoch", "spin_angle", "off_angle"), hae_longitude),
            "hae_latitude": (("epoch", "spin_angle", "off_angle"), hae_latitude),
        },
        coords={
            "epoch": [8.1794907049e17],
            "esa_energy_step": ESA_ENERGY_STEPS,
            "spin_angle": SPIN_ANGLE_BIN_CENTERS,
            "off_angle": OFF_ANGLE_BIN_CENTERS,
        },
    )
    return dataset


@pytest.fixture
def map():
    fake_field = np.zeros((1, 7, 60, 30))
    exposure_time = np.full((1, 7, 60, 30), 0.5)

    dataset = xr.Dataset(
        {
            "h_counts": (("epoch", "energy", "longitude", "latitude"), fake_field),
            "exposure_time": (
                ("epoch", "energy", "longitude", "latitude"),
                exposure_time,
            ),
            "h_rate": (("epoch", "energy", "longitude", "latitude"), fake_field),
            "h_flux": (("epoch", "energy", "longitude", "latitude"), fake_field),
            "solid_angle": (("epoch", "energy", "longitude", "latitude"), fake_field),
        },
        coords={
            "epoch": [8.1794907049e17],
            "longitude": [i for i in range(60)],
            "latitude": [i for i in range(30)],
            "energy": ESA_ENERGY_STEPS,
        },
    )
    return dataset


@pytest.fixture
def attr_mgr():
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="enamaps", level="l2-common")
    attr_mgr.add_instrument_variable_attrs(instrument="enamaps", level="l2-rectangular")
    return attr_mgr


@pytest.mark.external_kernel
def test_project_pset_to_rect_map(pset, imap_ena_sim_metakernel):
    # Arrange
    descriptor = "l090-ena-h-sf-nsp-ram-hae-6deg-3mo"

    # Act
    lo_rect_map = project_pset_to_sky_map([pset], descriptor)

    # Assert
    assert isinstance(lo_rect_map, RectangularSkyMap)
    assert lo_rect_map.spacing_deg == 6
    assert lo_rect_map.spice_reference_frame == geometry.SpiceFrame.IMAP_HAE
    assert lo_rect_map.num_points == 1800


@pytest.mark.external_kernel
def test_project_pset_to_healpix_map(pset, furnish_kernels):
    # Arrange
    descriptor = "l090-ena-h-sf-nsp-ram-hnu-nside2-3mo"
    kernels = [
        "imap_sclk_0000.tsc",
        "imap_science_100.tf",
        "naif0012.tls",
        "imap_spk_demo.bsp",
        "sim_1yr_imap_pointing_frame.bc",
    ]
    with furnish_kernels(kernels):
        # Act
        lo_rect_map = project_pset_to_sky_map([pset], descriptor)

    # Assert
    assert isinstance(lo_rect_map, HealpixSkyMap)
    assert lo_rect_map.nside == 2
    assert lo_rect_map.spice_reference_frame == geometry.SpiceFrame.IMAP_HNU
    assert lo_rect_map.num_points == 48


@pytest.mark.external_kernel
def test_calculate_rates(imap_ena_sim_metakernel):
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
def test_lo_l2(pset, imap_ena_sim_metakernel):
    # Arrange
    pset = {"imap_lo_l1c_pset": [pset]}
    descriptor = "l090-ena-h-sf-nsp-ram-hae-6deg-3mo"

    # Act
    hflux_map = lo_l2(pset, [], descriptor)

    # Assert
    assert len(hflux_map) == 1
    assert (
        hflux_map[0].attrs["Logical_source"]
        == "imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-3mo"
    )

    data_vars = ["h_counts", "exposure_time", "h_rate", "h_flux", "solid_angle"]
    for var in data_vars:
        assert var in hflux_map[0].data_vars, f"Variable {var} not found in dataset"
    assert hflux_map[0].data_vars["h_rate"].shape == (1, 7, 60, 30)


def test_add_attributes(map, attr_mgr):
    # Arrange
    logical_source = "imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-3mo"

    map_fields = {
        "epoch": "epoch",
        "h_flux": "ena_intensity",
        "h_rate": "ena_rate",
        "h_counts": "ena_count",
        "exposure_time": "exposure_factor",
        "energy": "energy",
        "solid_angle": "solid_angle",
        "longitude": "longitude",
        "latitude": "latitude",
    }

    # Act
    updated_map = add_attributes(map, attr_mgr, logical_source)

    # Assert
    for field, attr_name in map_fields.items():
        assert updated_map[field].attrs == attr_mgr.get_variable_attributes(
            attr_name, check_schema=False
        )

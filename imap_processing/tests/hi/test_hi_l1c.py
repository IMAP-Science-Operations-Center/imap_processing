"""Test coverage for imap_processing.hi.l1c.hi_l1c.py"""

from unittest import mock

import numpy as np
import pytest

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.hi.l1c import hi_l1c
from imap_processing.hi.utils import HIAPID


@pytest.mark.external_kernel()
@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
def test_generate_pset_dataset(hi_l1_test_data_path):
    """Test coverage for generate_pset_dataset function"""
    l1b_de_path = hi_l1_test_data_path / "imap_hi_l1b_45sensor-de_20250415_v999.cdf"
    l1b_dataset = load_cdf(l1b_de_path)
    l1c_dataset = hi_l1c.generate_pset_dataset(l1b_dataset)

    assert l1c_dataset.epoch.data[0] == np.mean(l1b_dataset.epoch.data[[0, -1]]).astype(
        np.int64
    )

    np.testing.assert_array_equal(l1c_dataset.despun_z.data.shape, (1, 3))
    np.testing.assert_array_equal(l1c_dataset.hae_latitude.data.shape, (1, 3600))
    np.testing.assert_array_equal(l1c_dataset.hae_longitude.data.shape, (1, 3600))
    for var in [
        "counts",
        "exposure_times",
        "background_rates",
        "background_rates_uncertainty",
    ]:
        np.testing.assert_array_equal(l1c_dataset[var].data.shape, (1, 9, 5, 3600))

    # Test ISTP compliance by writing CDF
    l1c_dataset.attrs["Data_version"] = 1
    write_cdf(l1c_dataset)


def test_empty_pset_dataset():
    """Test coverage for empty_pset_dataset function"""
    n_esa_steps = 9
    n_calibration_prods = 5
    sensor_str = HIAPID.H90_SCI_DE.sensor
    dataset = hi_l1c.empty_pset_dataset(n_esa_steps, sensor_str)

    assert dataset.epoch.size == 1
    assert dataset.spin_angle_bin.size == 3600
    assert dataset.esa_energy_step.size == n_esa_steps
    assert dataset.calibration_prod.size == n_calibration_prods

    # verify that attrs defined in hi_pset_epoch have overwritten default
    # epoch attributes
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs("hi")
    attr_mgr.add_instrument_variable_attrs(instrument="hi", level=None)
    pset_epoch_attrs = attr_mgr.get_variable_attributes(
        "hi_pset_epoch", check_schema=False
    )
    for k, v in pset_epoch_attrs.items():
        assert k in dataset.epoch.attrs
        assert dataset.epoch.attrs[k] == v


@pytest.mark.parametrize("sensor_str", ["90sensor", "45sensor"])
@mock.patch("imap_processing.spice.geometry.frame_transform")
@mock.patch("imap_processing.hi.l1c.hi_l1c.frame_transform")
def test_pset_geometry(mock_frame_transform, mock_geom_frame_transform, sensor_str):
    """Test coverage for pset_geometry function"""
    # pset_geometry uses both frame_transform and frame_transform_az_el. By mocking
    # the frame_transform imported into hi_l1c as well as the geometry.frame_transform
    # the underlying need for SPICE kernels is remove. Mock them both to just return
    # the input position vectors.
    mock_frame_transform.side_effect = lambda et, pos, from_frame, to_frame: pos
    mock_geom_frame_transform.side_effect = lambda et, pos, from_frame, to_frame: pos

    geometry_vars = hi_l1c.pset_geometry(0, sensor_str)

    assert "despun_z" in geometry_vars
    np.testing.assert_array_equal(geometry_vars["despun_z"].data, [[0, 0, 1]])

    assert "hae_latitude" in geometry_vars
    assert "hae_longitude" in geometry_vars
    # frame_transform is mocked to return the input vectors. For Hi-90, we
    # expect hae_latitude to be 0, and for Hi-45 we expect -45. Both sensors
    # have an expected longitude to be 0.1 degree steps starting at 0.05
    expected_latitude = 0 if sensor_str == "90sensor" else -45
    np.testing.assert_array_equal(
        geometry_vars["hae_latitude"].data, np.full((1, 3600), expected_latitude)
    )
    np.testing.assert_allclose(
        geometry_vars["hae_longitude"].data,
        np.arange(0.05, 360, 0.1, dtype=np.float32).reshape((1, 3600)),
        atol=4e-05,
    )

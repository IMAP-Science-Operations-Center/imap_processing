"""Test coverage for imap_processing.hi.l1c.hi_l1c.py"""

from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.hi.l1c import hi_l1c
from imap_processing.hi.l1c.hi_l1c import CalibrationProductConfig
from imap_processing.hi.utils import HIAPID


@pytest.fixture(scope="module")
def hi_test_cal_prod_config_path(hi_l1_test_data_path):
    return (
        hi_l1_test_data_path / "imap_his_pset-calibration-prod-config_20240101_v001.csv"
    )


@mock.patch("imap_processing.hi.l1c.hi_l1c.generate_pset_dataset")
def test_hi_l1c(mock_generate_pset_dataset, hi_test_cal_prod_config_path):
    """Test coverage for hi_l1c function"""
    mock_generate_pset_dataset.return_value = xr.Dataset(attrs={"Data_version": None})
    pset = hi_l1c.hi_l1c(
        [xr.Dataset(), hi_test_cal_prod_config_path], data_version="99"
    )
    assert pset.attrs["Data_version"] == "99"


def test_hi_l1c_not_implemented():
    """Test coverage for hi_l1c function with unrecognized dependencies"""
    with pytest.raises(NotImplementedError):
        hi_l1c.hi_l1c([None, None], "0")


@pytest.mark.external_kernel()
@pytest.mark.use_test_metakernel("imap_ena_sim_metakernel.template")
def test_generate_pset_dataset(hi_l1_test_data_path, hi_test_cal_prod_config_path):
    """Test coverage for generate_pset_dataset function"""
    l1b_de_path = hi_l1_test_data_path / "imap_hi_l1b_45sensor-de_20250415_v999.cdf"
    l1b_dataset = load_cdf(l1b_de_path)
    l1c_dataset = hi_l1c.generate_pset_dataset(
        l1b_dataset, hi_test_cal_prod_config_path
    )

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
        np.testing.assert_array_equal(l1c_dataset[var].data.shape, (1, 9, 2, 3600))

    # Test ISTP compliance by writing CDF
    l1c_dataset.attrs["Data_version"] = 1
    write_cdf(l1c_dataset)


def test_empty_pset_dataset():
    """Test coverage for empty_pset_dataset function"""
    n_energy_steps = 8
    l1b_esa_energy_steps = np.arange(n_energy_steps + 1).repeat(2)
    n_calibration_prods = 5
    sensor_str = HIAPID.H90_SCI_DE.sensor
    dataset = hi_l1c.empty_pset_dataset(
        l1b_esa_energy_steps, n_calibration_prods, sensor_str
    )

    assert dataset.epoch.size == 1
    assert dataset.spin_angle_bin.size == 3600
    assert dataset.esa_energy_step.size == n_energy_steps
    np.testing.assert_array_equal(
        dataset.esa_energy_step.data, np.arange(n_energy_steps) + 1
    )
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


class TestCalibrationProductConfig:
    """
    All test coverage for the pd.DataFrame accessor extension "cal_prod_config".
    """

    def test_wrong_columns(self):
        """Test coverage for a dataframe with the wrong columns."""
        required_columns = CalibrationProductConfig.required_columns
        for exclude_column_name in required_columns:
            include_columns = set(required_columns) - {exclude_column_name}
            df = pd.DataFrame({col: [1, 2, 3] for col in include_columns})
            with pytest.raises(AttributeError, match="Required column*"):
                _ = df.cal_prod_config.number_of_products

    def test_from_csv(self, hi_test_cal_prod_config_path):
        """Test coverage for read_csv function."""
        df = CalibrationProductConfig.from_csv(hi_test_cal_prod_config_path)
        assert isinstance(df["coincidence_type_list"][0, 1], list)

    def test_number_of_products(self, hi_test_cal_prod_config_path):
        """Test coverage for number of products accessor."""
        df = CalibrationProductConfig.from_csv(hi_test_cal_prod_config_path)
        assert df.cal_prod_config.number_of_products == 2

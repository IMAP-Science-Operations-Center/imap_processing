"""Test coverage for imap_processing.hi.l1c.hi_l1c.py"""

import numpy as np

from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.hi.l1c import hi_l1c
from imap_processing.hi.utils import HIAPID


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

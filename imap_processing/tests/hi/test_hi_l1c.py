"""Test coverage for imap_processing.hi.l1c.hi_l1c.py"""

import numpy as np

from imap_processing.hi.l1a.hi_l1a import hi_l1a
from imap_processing.hi.l1b.hi_l1b import hi_l1b
from imap_processing.hi.l1c import hi_l1c
from imap_processing.hi.utils import HIAPID


def test_generate_pset_dataset(create_de_data):
    """Test coverage for generate_pset_dataset function"""
    # TODO: once things are more stable, check in an L1B DE file as test data?
    # For now, test using false de data run through l1a and l1b processing
    bin_data_path = create_de_data(HIAPID.H45_SCI_DE.value)
    processed_data = hi_l1a(bin_data_path, "002")
    l1b_dataset = hi_l1b(processed_data[0], "002")

    l1c_dataset = hi_l1c.generate_pset_dataset(l1b_dataset)

    assert l1c_dataset.epoch.data[0] == l1b_dataset.epoch.data[0]


def test_allocate_pset_dataset():
    """Test coverage for allocate_pset_dataset function"""
    n_esa_steps = 10
    sensor_str = HIAPID.H90_SCI_DE.sensor
    dataset = hi_l1c.allocate_pset_dataset(n_esa_steps, sensor_str)

    assert dataset.epoch.size == 1
    assert dataset.spin_angle_bin.size == 3600
    np.testing.assert_array_equal(dataset.despun_z.data.shape, (1, 3))
    np.testing.assert_array_equal(dataset.hae_latitude.data.shape, (1, 3600))
    np.testing.assert_array_equal(dataset.hae_longitude.data.shape, (1, 3600))
    n_esa_step = dataset.esa_energy_step.data.size
    for var in [
        "counts",
        "exposure_times",
        "background_rates",
        "background_rates_uncertainty",
    ]:
        np.testing.assert_array_equal(dataset[var].data.shape, (1, n_esa_step, 3600))

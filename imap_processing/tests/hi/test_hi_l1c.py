"""Test coverage for imap_processing.hi.l1c.hi_l1c.py"""

import numpy as np

from imap_processing.cdf.utils import write_cdf
from imap_processing.hi.l1a.hi_l1a import hi_l1a
from imap_processing.hi.l1b.hi_l1b import hi_l1b
from imap_processing.hi.l1c.hi_l1c import hi_l1c
from imap_processing.hi.utils import HIAPID


def test_hi_l1c_pset(create_de_data):
    """Test coverage for hi_l1c function"""
    # TODO: once things are more stable, check in an L1B DE file as test data?
    # For now, test using false de data run through l1a and l1b processing
    bin_data_path = create_de_data(HIAPID.H45_SCI_DE.value)
    processed_data = hi_l1a(packet_file_path=bin_data_path)
    l1a_cdf_path = write_cdf(processed_data[0])
    l1b_dataset = hi_l1b(l1a_cdf_path)
    l1b_cdf_path = write_cdf(l1b_dataset)

    l1c_dataset = hi_l1c(l1b_cdf_path, "002")

    assert l1c_dataset.epoch.size == 1
    assert l1c_dataset.spin_angle_bin.size == 3600
    np.testing.assert_array_equal(l1c_dataset.despun_z.data.shape, (1, 3))
    np.testing.assert_array_equal(l1c_dataset.hae_latitude.data.shape, (1, 3600))
    np.testing.assert_array_equal(l1c_dataset.hae_longitude.data.shape, (1, 3600))
    n_esa_step = l1c_dataset.esa_step.data.size
    for var in [
        "counts",
        "exposure_times",
        "background_rates",
        "background_rates_uncertainty",
    ]:
        np.testing.assert_array_equal(
            l1c_dataset[var].data.shape, (1, n_esa_step, 3600)
        )

"""Test coverage for imap_processing.hi.l2.hi_l2.py"""

import numpy as np
import pytest

from imap_processing.cdf.utils import load_cdf
from imap_processing.hi.l2.hi_l2 import HiPointingSet
from imap_processing.spice.geometry import SpiceFrame


@pytest.fixture(scope="module")
def pset_path(hi_l1_test_data_path):
    return hi_l1_test_data_path / "imap_hi_l1c_45sensor-pset_20250415_v999.cdf"


@pytest.mark.external_test_data
class TestHiPointingSet:
    """Test suite for HiPointingSet class."""

    def test_init(self, pset_path):
        """Test coverage for __init__ method."""
        pset_ds = load_cdf(pset_path)
        hi_pset = HiPointingSet(pset_ds)
        assert isinstance(hi_pset, HiPointingSet)
        assert hi_pset.spice_reference_frame == SpiceFrame.ECLIPJ2000
        assert hi_pset.num_points == 3600
        np.testing.assert_array_equal(hi_pset.az_el_points.shape, (3600, 2))

    def test_from_cdf(self, pset_path):
        """Test coverage for from_cdf method."""
        hi_pset = HiPointingSet.from_cdf(pset_path)
        assert isinstance(hi_pset, HiPointingSet)

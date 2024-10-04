"""Test coverage for imap_processing.hi.l1b.hi_l1b.py"""

from imap_processing.hi.l1a.hi_l1a import hi_l1a
from imap_processing.hi.l1b.hi_l1b import hi_l1b
from imap_processing.hi.utils import HIAPID


def test_hi_l1b_hk(hi_l0_test_data_path):
    """Test coverage for imap_processing.hi.hi_l1b.hi_l1b() with
    housekeeping L1A as input"""
    # TODO: once things are more stable, check in an L1A HK file as test data
    bin_data_path = hi_l0_test_data_path / "20231030_H45_APP_NHK.bin"
    data_version = "001"
    processed_data = hi_l1a(packet_file_path=bin_data_path, data_version=data_version)

    l1b_dataset = hi_l1b(processed_data[0], data_version=data_version)
    assert l1b_dataset.attrs["Logical_source"] == "imap_hi_l1b_45sensor-hk"


def test_hi_l1b_de(create_de_data, tmp_path):
    """Test coverage for imap_processing.hi.hi_l1b.hi_l1b() with
    direct events L1A as input"""
    # TODO: once things are more stable, check in an L1A DE file as test data
    # Process using test data
    bin_data_path = create_de_data(HIAPID.H45_SCI_DE.value)
    data_version = "001"
    processed_data = hi_l1a(packet_file_path=bin_data_path, data_version=data_version)

    l1b_dataset = hi_l1b(processed_data[0], data_version=data_version)
    assert l1b_dataset.attrs["Logical_source"] == "imap_hi_l1b_45sensor-de"
    assert len(l1b_dataset.data_vars) == 14

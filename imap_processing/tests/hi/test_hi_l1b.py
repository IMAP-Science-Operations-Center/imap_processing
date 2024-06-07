"""Test coverage for imap_processing.hi.hi_l1b.py"""

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import write_cdf
from imap_processing.hi.hi_cdf_attrs import hi_hk_l1b_global_attrs
from imap_processing.hi.l1a.hi_l1a import hi_l1a
from imap_processing.hi.l1b.hi_l1b import hi_l1b


def test_hi_l1b_hk():
    """Test coverage for imap_processing.hi.hi_l1b.hi_l1b() with
    housekeeping L1A as input"""
    # TODO: once things are more stable, check in an L1A HK file as test data
    test_path = imap_module_directory / "tests/hi/l0_test_data"
    bin_data_path = test_path / "20231030_H45_APP_NHK.bin"
    processed_data = hi_l1a(packet_file_path=bin_data_path)
    l1a_cdf_path = write_cdf(processed_data[0])

    l1b_dataset = hi_l1b(l1a_cdf_path)
    assert l1b_dataset.attrs["Logical_source"] == hi_hk_l1b_global_attrs.logical_source


def test_hi_l1b_de(create_de_data, tmp_path):
    """Test coverage for imap_processing.hi.hi_l1b.hi_l1b() with
    direct events L1A as input"""
    # TODO: once things are more stable, check in an L1A DE file as test data
    # Process using test data
    bin_data_path = tmp_path / "imap_hi_l0_sdc-test-data_20240318_v000.pkts"
    processed_data = hi_l1a(packet_file_path=bin_data_path)
    l1a_cdf_path = write_cdf(processed_data[0])

    l1b_dataset = hi_l1b(l1a_cdf_path)
    assert l1b_dataset.attrs["Logical_source"] == "imap_hi_l1b_45sensor-de"
    assert len(l1b_dataset.data_vars) == 14

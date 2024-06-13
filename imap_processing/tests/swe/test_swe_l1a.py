from imap_processing import imap_module_directory
from imap_processing.cdf.utils import write_cdf
from imap_processing.swe.l1a.swe_l1a import swe_l1a
from imap_processing.swe.utils.swe_utils import (
    SWEAPID,
)
from imap_processing.utils import group_by_apid


def test_group_by_apid(decom_test_data):
    grouped_data = group_by_apid(decom_test_data)

    # check total dataset for swe science
    total_science_data = grouped_data[SWEAPID.SWE_SCIENCE]
    assert len(total_science_data) == 29


def test_cdf_creation():
    test_data_path = "tests/swe/l0_data/2024051010_SWE_SCIENCE_packet.bin"
    processed_data = swe_l1a(imap_module_directory / test_data_path, "001")

    cem_raw_cdf_filepath = write_cdf(processed_data)

    assert cem_raw_cdf_filepath.name == "imap_swe_l1a_sci_20240510_v001.cdf"

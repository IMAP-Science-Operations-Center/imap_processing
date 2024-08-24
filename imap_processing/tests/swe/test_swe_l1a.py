from imap_processing import imap_module_directory
from imap_processing.cdf.utils import write_cdf
from imap_processing.swe.l1a.swe_l1a import swe_l1a


def test_cdf_creation():
    test_data_path = "tests/swe/l0_data/2024051010_SWE_SCIENCE_packet.bin"
    processed_data = swe_l1a(imap_module_directory / test_data_path, "001")

    cem_raw_cdf_filepath = write_cdf(processed_data)

    assert cem_raw_cdf_filepath.name == "imap_swe_l1a_sci_20240510_v001.cdf"

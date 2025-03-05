from imap_processing import imap_module_directory
from imap_processing.cdf.utils import write_cdf
from imap_processing.swe.l1a.swe_l1a import swe_l1a


def test_cdf_creation():
    test_data_path = "tests/swe/l0_data/2024051010_SWE_SCIENCE_packet.bin"
    processed_data = swe_l1a(imap_module_directory / test_data_path, "001")

    cem_raw_cdf_filepath = write_cdf(processed_data[0])

    assert cem_raw_cdf_filepath.name == "imap_swe_l1a_sci_20240510_v001.cdf"


def test_cdf_creation_hk():
    test_data_path = "tests/swe/l0_data/2024051010_SWE_HK_packet.bin"
    processed_data = swe_l1a(imap_module_directory / test_data_path, "001")

    hk_cdf_filepath = write_cdf(processed_data[0])

    assert hk_cdf_filepath.name == "imap_swe_l1a_hk_20240510_v001.cdf"


def test_cdf_creation_cem_raw():
    test_data_path = "tests/swe/l0_data/2024051011_SWE_CEM_RAW_packet.bin"
    processed_data = swe_l1a(imap_module_directory / test_data_path, "001")

    cem_raw_cdf_filepath = write_cdf(processed_data[0])

    assert cem_raw_cdf_filepath.name == "imap_swe_l1a_cem-raw_20240510_v001.cdf"

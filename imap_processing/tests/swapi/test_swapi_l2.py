from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.swapi.l1.swapi_l1 import swapi_l1
from imap_processing.swapi.l2.swapi_l2 import swapi_l2
from imap_processing.swapi.swapi_utils import SWAPIAPID


def test_swapi_l1_cdf():
    """Test housekeeping processing and CDF file creation"""
    l0_data_path = (
        f"{imap_module_directory}/tests/swapi/l0_data/"
        "imap_swapi_l0_raw_20231012_v001.pkts"
    )
    processed_data = swapi_l1(l0_data_path, data_version="v001")

    # sci cdf file
    cdf_filename = "imap_swapi_l1_sci-1min_20100101_v001.cdf"

    cdf_path = write_cdf(processed_data[0])
    assert cdf_path.name == cdf_filename

    # process science data to l2
    l1_dataset = load_cdf(cdf_path)
    assert l1_dataset.attrs["Apid"] == f"{SWAPIAPID.SWP_SCI}"
    assert l1_dataset.attrs["Plan_id"] == "0"
    assert l1_dataset.attrs["Sweep_table"] == "0"

    # print("after ready l1 file ", l1_dataset.attrs.keys())
    l2_processed_data = swapi_l2(l1_dataset, data_version="v001")
    l2_cdf = write_cdf(l2_processed_data)
    assert l2_cdf.name == "imap_swapi_l2_sci-1min_20100101_v001.cdf"

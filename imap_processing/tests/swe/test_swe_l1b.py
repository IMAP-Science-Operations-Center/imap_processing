import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.swe.l1a.swe_l1a import swe_l1a
from imap_processing.swe.l1a.swe_science import swe_science
from imap_processing.swe.l1b.swe_l1b import swe_l1b
from imap_processing.swe.utils.swe_utils import (
    SWEAPID,
)
from imap_processing.utils import convert_raw_to_eu, group_by_apid


def test_swe_l1b(decom_test_data):
    """Test that calculate engineering unit(EU) is
    matches validation data.

    Parameters
    ----------
    decom_test_data : list
        List of packets
    """
    grouped_data = group_by_apid(decom_test_data)
    # Process science to l1a.
    # because of test data being in the wrong
    # order, we need to manually re-sort data
    # into order.
    sorted_packets = sorted(
        grouped_data[SWEAPID.SWE_SCIENCE],
        key=lambda x: x.data["QUARTER_CYCLE"].raw_value,
    )
    science_l1a_ds = swe_science(sorted_packets, "001")
    # convert value from raw to engineering units as needed
    conversion_table_path = (
        imap_module_directory / "swe/l1b/engineering_unit_convert_table.csv"
    )
    # Look up packet name from APID
    packet_name = SWEAPID.SWE_SCIENCE.name
    # Convert raw data to engineering units as needed
    science_l1b = convert_raw_to_eu(
        science_l1a_ds,
        conversion_table_path=conversion_table_path,
        packet_name=packet_name,
    )

    # read science validation data
    test_data_path = imap_module_directory / "tests/swe/l0_validation_data"
    eu_validation_data = pd.read_csv(
        test_data_path / "idle_export_eu.SWE_SCIENCE_20240510_092742.csv",
        index_col="SHCOARSE",
    )

    second_data = sorted_packets[1]
    validation_data = eu_validation_data.loc[second_data.data["SHCOARSE"].raw_value]

    science_eu_field_list = [
        "SPIN_PHASE",
        "SPIN_PERIOD",
        "THRESHOLD_DAC",
    ]

    # Test EU values for science data
    for field in science_eu_field_list:
        assert round(science_l1b[field.lower()].data[1], 5) == round(
            validation_data[field], 5
        )


@pytest.mark.xfail(reason="L1B requires updates")
def test_cdf_creation():
    """Test that CDF file is created and has the correct name."""
    test_data_path = "tests/swe/l0_data/2024051010_SWE_SCIENCE_packet.bin"
    l1a_datasets = swe_l1a(imap_module_directory / test_data_path, "002")

    sci_l1a_filepath = write_cdf(l1a_datasets)

    assert sci_l1a_filepath.name == "imap_swe_l1a_sci_20240510_v002.cdf"

    # reads data from CDF file and passes to l1b
    l1a_cdf_dataset = load_cdf(sci_l1a_filepath)
    l1b_dataset = swe_l1b(l1a_cdf_dataset, "002")

    sci_l1b_filepath = write_cdf(l1b_dataset)

    assert sci_l1b_filepath.name == "imap_swe_l1b_sci_20240510_v002.cdf"

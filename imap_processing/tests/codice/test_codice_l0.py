"""Tests the decommutation process for CoDICE CCSDS Packets. This also tests the
unit conversion process for CoDICE housekeeping data."""

from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.codice import codice_l0
from imap_processing.codice.codice_l1a import create_hskp_dataset
from imap_processing.utils import convert_raw_to_eu

# Define the CCSDS header fields (which will be ignored in these tests)
CCSDS_HEADER_FIELDS = [
    "shcoarse",
    "version",
    "type",
    "sec_hdr_flg",
    "pkt_apid",
    "seq_flgs",
    "src_seq_ctr",
    "pkt_len",
]


@pytest.fixture(scope="session")
def decom_test_data() -> xr.Dataset:
    """Read test data from file and return a decommutated housekeeping packet.

    Returns
    -------
    packet : xr.Dataset
        A decommutated housekeeping packet
    """

    packet_file = Path(
        f"{imap_module_directory}/tests/codice/data/"
        f"imap_codice_l0_raw_20241110_v001.pkts"
    )

    packet = codice_l0.decom_packets(packet_file)[1136]

    return packet


@pytest.fixture(scope="session")
def validation_data() -> pd.core.frame.DataFrame:
    """Read in validation data from the CSV file

    Returns
    -------
    validation_data : pandas.core.frame.DataFrame
        The validation data read from the CSV, cleaned up and ready to compare
        the decommutated packet with
    """

    # Read in the CSV file
    validation_file = Path(
        f"{imap_module_directory}/tests/codice/data/"
        f"idle_export_raw.COD_NHK_20230822_122700.csv"
    )
    validation_data = pd.read_csv(validation_file, index_col="SHCOARSE")

    if "timestamp" in validation_data.columns:
        validation_data.drop(columns=["timestamp"], inplace=True, errors="ignore")

    return validation_data


@pytest.mark.xfail(
    reason="Need to update to validate against new validation CDFs. See issue #1154."
)
def test_eu_hskp_data(
    decom_test_data: xr.Dataset,
    validation_data: pd.core.frame.DataFrame,
):
    """Compare the engineering unit (EU) housekeeping data to the validation data.

    Parameters
    ----------
    decom_test_data : xr.Dataset
        The decommutated housekeeping packet
    validation_data : pandas.core.frame.DataFrame
        The validation data to compare against
    """

    l1a_hk_ds = create_hskp_dataset(decom_test_data, "001")
    eu_hk_data = convert_raw_to_eu(
        l1a_hk_ds,
        imap_module_directory / "tests/codice/data/eu_unit_lookup_table.csv",
        "P_COD_NHK",
    )

    validation_row = validation_data.loc[decom_test_data.shcoarse]

    # Compare EU values of housekeeping data, skipping CCSDS header fields
    for field in eu_hk_data:
        # Skip header values
        if field in CCSDS_HEADER_FIELDS:
            continue

        eu_values = getattr(eu_hk_data, field).data
        validation_values = validation_row[field.upper()]

        # Compare each individual element
        for eu_val, validation_val in zip(eu_values, validation_values):
            assert round(eu_val, 5) == round(validation_val, 5)


@pytest.mark.xfail(
    reason="Need to update to validate against new validation CDFs. See issue #1154."
)
def test_raw_hskp_data(
    decom_test_data: xr.Dataset,
    validation_data: pd.core.frame.DataFrame,
):
    """Compare the raw housekeeping data to the validation data.

    Parameters
    ----------
    decom_test_data : xr.Dataset
        The decommutated housekeeping packet
    validation_data : pandas.core.frame.DataFrame
        The validation data to compare against
    """

    validation_row = validation_data.loc[decom_test_data.shcoarse]

    # Compare raw values of housekeeping data
    for field in decom_test_data:
        if field not in CCSDS_HEADER_FIELDS:
            raw_values = getattr(decom_test_data, field).data
            validation_values = validation_row[field.upper()]
            for raw_value, validation_value in zip(raw_values, validation_values):
                assert raw_value == validation_value


def test_total_packets_in_data_file(decom_test_data: xr.Dataset):
    """Test if total packets in data file is correct

    Parameters
    ----------
    decom_test_data : xr.Dataset
        The decommutated housekeeping packet
    """

    total_packets = 31778
    assert len(decom_test_data.epoch) == total_packets

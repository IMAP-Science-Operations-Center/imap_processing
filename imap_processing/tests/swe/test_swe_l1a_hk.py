from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.swe.utils.swe_utils import SWEAPID
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture(scope="session")
def decom_hk_data():
    """Read test data from file"""
    test_folder_path = imap_module_directory / "tests/swe/l0_data"
    packet_file = test_folder_path / "2024051010_SWE_HK_packet.bin"
    xtce_document = (
        imap_module_directory / "swe/packet_definitions/swe_packet_definition.xml"
    )
    datasets_by_apid = packet_file_to_datasets(
        packet_file,
        xtce_document,
        use_derived_value=True,
    )
    return datasets_by_apid[SWEAPID.SWE_APP_HK]


def test_swe_hk_data(decom_hk_data):
    """This test and validate raw data of SWAPI raw housekeeping data."""
    # read validation data
    test_data_path = imap_module_directory / "tests/swe/l0_validation_data"

    raw_validation_data = pd.read_csv(
        test_data_path / "idle_export_eu.SWE_APP_HK_20240510_092742.csv",
        index_col="SHCOARSE",
    )

    first_data = decom_hk_data.isel(epoch=0)
    validation_data = raw_validation_data.loc[first_data["shcoarse"].values]

    # TODO: check with SWE team
    mismatched_keys = [
        "HVPS_ESA_DAC",
    ]

    ccsds_header_keys = [field.name for field in fields(CcsdsData)]
    ccsds_header_keys += ["SHCOARSE"]
    # compare raw values of validation data
    for key in first_data.keys():
        data_key = key.upper()
        if data_key in ccsds_header_keys:
            continue

        if data_key in mismatched_keys:
            continue

        # Compare derived string values to validation data
        if str(first_data[key].dtype).startswith("<U"):
            assert first_data[key].values == validation_data[data_key]
        # Compare derived float values to validation data
        elif first_data[key].dtype == "float64":
            np.testing.assert_almost_equal(
                first_data[key].values, validation_data[data_key], decimal=6
            )
        # Compare derived integer values to validation data
        else:
            assert first_data[key].values == validation_data[data_key]

"""Tests coverage for imap_processing/utils.py"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import utils


def test_convert_raw_to_eu(tmp_path):
    """Test coverage for utils.convert_raw_to_eu()"""
    # Generate a csv for testing
    test_df = pd.DataFrame(
        data={
            "packetName": ["PACKET_0", "PACKET_0", "PACKET_1"],
            "mnemonic": ["FIELD_0", "FIELD_1", "FIELD_0"],
            "convertAs": ["UNSEGMENTED_POLY", "UNSEGMENTED_POLY", "FOO_METHOD"],
            "c0": [0, 1, 2],
            "c1": [0, 1, 2],
            "c2": [0, 0, 2],
            "c3": [0, 0, 2],
            "c4": [0, 0, 2],
            "c5": [0, 1, 2],
            "c6": [0, 0, 2],
            "c7": [0, 0, 2],
            "unit": ["a", "b", "c"],
        }
    )
    test_csv = tmp_path / "test_convert_table.csv"
    with open(test_csv, "w") as f:
        f.write("# Comment on first line of file\n")
        test_df.to_csv(f, index=False)

    # Generate a fake dataset for testing
    n_packets = 3
    field_0 = np.arange(n_packets)
    field_1 = np.arange(n_packets) + 10
    time = np.arange(n_packets) + 1000
    dn_dataset = xr.Dataset(
        data_vars=dict(
            FIELD_0=(["time"], field_0),
            FIELD_1=(["time"], field_1),
        ),
        coords=dict(
            time=time,
        ),
    )

    eu_dataset = utils.convert_raw_to_eu(
        dn_dataset.copy(), test_csv.absolute(), "PACKET_0", comment="#"
    )

    # Check the converted values by manually doing the polynomial math
    np.testing.assert_array_equal(eu_dataset["FIELD_0"].data, np.zeros(n_packets))
    assert eu_dataset["FIELD_0"].attrs["units"] == test_df["unit"].iloc[0]

    field_1_coeffs = (
        test_df[test_df["mnemonic"] == "FIELD_1"].filter(regex=r"c\d").values[0]
    )
    field_1_compare = np.zeros(n_packets)
    for p, coeff in enumerate(field_1_coeffs):
        field_1_compare += coeff * np.power(field_1, p)
    np.testing.assert_array_equal(eu_dataset["FIELD_1"].data, field_1_compare)

    # Check that a ValueError is raised for unexpected conversion specified in
    # conversion table "convertAs" column
    with pytest.raises(ValueError, match=r"Unexpected conversion type: .*"):
        eu_dataset = utils.convert_raw_to_eu(
            dn_dataset.copy(), test_csv.absolute(), "PACKET_1", comment="#"
        )

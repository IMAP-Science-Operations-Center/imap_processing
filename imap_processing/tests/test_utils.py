"""Tests coverage for imap_processing/utils.py"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory, utils


def test_convert_raw_to_eu(tmp_path):
    """Test coverage for utils.convert_raw_to_eu()"""
    # Generate a csv for testing
    # Include segmented and unsegmented polyvals
    dn_cutoff = 2
    test_df = pd.DataFrame(
        data={
            "packetName": ["PACKET_0", "PACKET_0", "PACKET_2", "PACKET_2", "PACKET_1"],
            "mnemonic": ["FIELD_0", "FIELD_1", "FIELD_2", "FIELD_2", "FIELD_0"],
            "convertAs": [
                "UNSEGMENTED_POLY",
                "UNSEGMENTED_POLY",
                "SEGMENTED_POLY",
                "SEGMENTED_POLY",
                "FOO_METHOD",
            ],
            "dn_range_start": [0, 0, 1, dn_cutoff + 1, 0],
            "dn_range_stop": [0, 0, dn_cutoff, 4, 0],
            "c0": [0, 1, 3, 4, 2],
            "c1": [0, 1, 3, 4, 2],
            "c2": [0, 0, 3, 4, 2],
            "c3": [0, 0, 3, 4, 2],
            "c5": [0, 1, 3, 4, 2],
            "c6": [0, 0, 3, 4, 2],
            "c7": [0, 0, 3, 4, 2],
            "unit": ["a", "b", "d", "d", "c"],
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
    field_2 = np.arange(n_packets) + 1
    time = np.arange(n_packets) + 1000
    dn_dataset = xr.Dataset(
        data_vars=dict(
            FIELD_0=(["time"], field_0),
            FIELD_1=(["time"], field_1),
            FIELD_2=(["time"], field_2),
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
    assert eu_dataset["FIELD_0"].attrs["UNITS"] == test_df["unit"].iloc[0]

    field_1_coeffs = (
        test_df[test_df["mnemonic"] == "FIELD_1"].filter(regex=r"c\d").values[0]
    )
    field_1_compare = np.zeros(n_packets)
    for p, coeff in enumerate(field_1_coeffs):
        field_1_compare += coeff * np.power(field_1, p)
    np.testing.assert_array_equal(eu_dataset["FIELD_1"].data, field_1_compare)

    eu_dataset = utils.convert_raw_to_eu(
        dn_dataset.copy(), test_csv.absolute(), "PACKET_2", comment="#"
    )
    # Test the segmented polynomials
    field_2_coeffs = (
        test_df[test_df["mnemonic"] == "FIELD_2"].filter(regex=r"c\d").values
    )
    field_2_compare_seg_1 = np.zeros(len(field_2[field_2 <= dn_cutoff]))
    field_2_compare_seg_2 = np.zeros(len(field_2[field_2 > dn_cutoff]))
    for p, coeff in enumerate(field_2_coeffs[0]):
        field_2_compare_seg_1 += coeff * np.power(field_2[field_2 <= dn_cutoff], p)
    for p, coeff in enumerate(field_2_coeffs[1]):
        field_2_compare_seg_2 += coeff * np.power(field_2[field_2 > dn_cutoff], p)

    field_2_compare = np.concatenate([field_2_compare_seg_1, field_2_compare_seg_2])
    np.testing.assert_array_equal(eu_dataset["FIELD_2"].data, field_2_compare)

    assert eu_dataset["FIELD_2"].attrs["UNITS"] == test_df["unit"].iloc[2]
    # Check that a ValueError is raised for unexpected conversion specified in
    # conversion table "convertAs" column
    with pytest.raises(
        ValueError,
        match="Column 'convertAs' must all be UNSEGMENTED_POLY or "
        "SEGMENTED_POLY for a packet name and mnemonic",
    ):
        utils.convert_raw_to_eu(
            dn_dataset.copy(), test_csv.absolute(), "PACKET_1", comment="#"
        )


def test_segmented_poly_out_of_range(tmp_path):
    """Test that a value error is thrown if a raw DN value is out of range"""
    # Generate a csv for testing
    test_df = pd.DataFrame(
        data={
            "packetName": ["PACKET_0", "PACKET_0"],
            "mnemonic": ["FIELD_0", "FIELD_0"],
            "convertAs": ["SEGMENTED_POLY", "SEGMENTED_POLY"],
            # Make ranges higher than DNs
            "dn_range_start": [100, 201],
            "dn_range_stop": [200, 400],
            "c0": [0, 1],
            "c1": [0, 1],
            "c2": [0, 1],
            "c3": [0, 1],
            "c5": [0, 1],
            "c6": [0, 1],
            "c7": [0, 1],
            "unit": ["a", "b"],
        }
    )
    test_csv = tmp_path / "test_convert_table.csv"
    with open(test_csv, "w") as f:
        f.write("# Comment on first line of file\n")
        test_df.to_csv(f, index=False)

    # Generate a fake dataset for testing
    n_packets = 3
    field_0 = np.arange(n_packets)
    time = np.arange(n_packets) + 1000
    dn_dataset = xr.Dataset(
        data_vars=dict(
            FIELD_0=(["time"], field_0),
        ),
        coords=dict(
            time=time,
        ),
    )
    # Check that a ValueError is raised for DNs not in any ranges
    with pytest.raises(
        ValueError, match="Raw DN values found outside of the expected range"
    ):
        utils.convert_raw_to_eu(
            dn_dataset.copy(), test_csv.absolute(), "PACKET_0", comment="#"
        )


def test_unsegmented_poly_multiple_rows(tmp_path):
    """Test that a value error is thrown if there are multiple rows with the same
    mnemonic and packet name and convertAs is UNSEGMENTED_POLY"""
    # Generate a csv for testing
    test_df = pd.DataFrame(
        data={
            "packetName": ["PACKET_0", "PACKET_0"],
            "mnemonic": ["FIELD_0", "FIELD_0"],
            "convertAs": ["UNSEGMENTED_POLY", "UNSEGMENTED_POLY"],
            "c0": [0, 1],
            "c1": [0, 1],
            "c2": [0, 1],
            "c3": [0, 1],
            "c5": [0, 1],
            "c6": [0, 1],
            "c7": [0, 1],
            "unit": ["a", "a"],
        }
    )
    test_csv = tmp_path / "test_convert_table.csv"
    with open(test_csv, "w") as f:
        f.write("# Comment on first line of file\n")
        test_df.to_csv(f, index=False)

    # Generate a fake dataset for testing
    field_0 = np.arange(3)
    dn_dataset = xr.Dataset(
        data_vars=dict(
            FIELD_0=(["time"], field_0),
        )
    )
    # Check that a ValueError is raised for DNs not in any ranges
    with pytest.raises(
        ValueError,
        match="For unsegmented polynomial conversions, there should "
        "only be one row per mnemonic and packet name",
    ):
        utils.convert_raw_to_eu(
            dn_dataset.copy(), test_csv.absolute(), "PACKET_0", comment="#"
        )


@pytest.mark.parametrize(
    "use_derived_value, expected_mode",
    [(True, np.array(["HVENG", "HVSCI"])), (False, np.array([2, 3]))],
)
def test_packet_file_to_datasets(use_derived_value, expected_mode):
    """
    Test that all datatypes aren't all int64 and that we get
    uint8/uint16 from header items as expected.

    Test that we get multiple apids in the output.
    """
    test_file = "tests/swapi/l0_data/imap_swapi_l0_raw_20240924_v001.pkts"
    packet_files = imap_module_directory / test_file
    packet_definition = (
        imap_module_directory / "swapi/packet_definitions/swapi_packet_definition.xml"
    )
    datasets_by_apid = utils.packet_file_to_datasets(
        packet_files, packet_definition, use_derived_value=use_derived_value
    )
    # 2 apids in the SWAPI test data that we decommutate
    # (2 others are not included in the XTCE definition, but are in the raw packet file)
    assert len(datasets_by_apid) == 2
    data = datasets_by_apid[1188]
    assert data["sec_hdr_flg"].dtype == np.uint8
    assert data["pkt_apid"].dtype == np.uint16
    np.testing.assert_array_equal(np.unique(data["mode"].data), expected_mode)


def test_packet_file_to_datasets_flat_definition():
    test_file = "tests/idex/test_data/imap_idex_l0_raw_20231218_v001.pkts"
    packet_files = imap_module_directory / test_file
    packet_definition = (
        imap_module_directory / "idex/packet_definitions/idex_packet_definition.xml"
    )
    with pytest.raises(ValueError, match="Packet fields do not match"):
        utils.packet_file_to_datasets(packet_files, packet_definition)

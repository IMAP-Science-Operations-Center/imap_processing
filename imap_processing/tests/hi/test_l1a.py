import numpy as np
import pandas as pd

from imap_processing.cdf.utils import write_cdf
from imap_processing.hi.hi_l1a import (
    create_de_dataset,
    hi_l1a,
    parse_direct_events,
    unpack_hist_counter,
)
from imap_processing.hi.utils import HIAPID

VALIDATION_TO_TEST_COLUMN_MAP = {
    "PHVERNO": "version",
    "PHTYPE": "type",
    "PHSHF": "sec_hdr_flg",
    "PHAPID": "pkt_apid",
    "PHGROUPF": "seq_flgs",
    "PHSEQCNT": "src_seq_ctr",
    "PHDLEN": "pkt_len",
}


def test_sci_de_decom(hi_l0_test_data_path):
    """Test science direct event data"""

    bin_data_path = hi_l0_test_data_path / "H90_sci_de_20241104.bin"
    processed_data = hi_l1a(bin_data_path)

    assert processed_data[0].attrs["Logical_source"] == "imap_hi_l1a_90sensor-de"

    # TODO: Verify correct unpacking of sample data. Issue: #1186

    # Write to CDF
    cdf_filename = "imap_hi_l1a_90sensor-de_20241105_v999.cdf"
    cdf_filepath = write_cdf(processed_data[0])
    assert cdf_filepath.name == cdf_filename


def test_diag_fee_decom(hi_l0_test_data_path):
    """Test diag_fee data"""
    bin_data_path = hi_l0_test_data_path / "H45_diag_fee_20250208.bin"
    processed_data = hi_l1a(packet_file_path=bin_data_path)
    dataset = processed_data[0]
    cdf_filepath = write_cdf(processed_data[0], istp=False)
    assert cdf_filepath.name == "imap_hi_l1a_45sensor-diagfee_20250208_v999.cdf"

    assert np.unique(processed_data[0]["pkt_apid"].values) == HIAPID.H45_DIAG_FEE.value

    validation_df = pd.read_csv(
        hi_l0_test_data_path / "H45_diag_fee_20250208_verify.csv"
    )

    for col_name, series in validation_df.items():
        if col_name == "timestamp":
            continue
        ds_var_name = VALIDATION_TO_TEST_COLUMN_MAP.get(col_name, col_name.lower())
        np.testing.assert_array_equal(series.values, dataset[ds_var_name].data)


def test_app_nhk_decom(hi_l0_test_data_path):
    """Test housekeeping data"""

    # Unpack housekeeping data
    bin_data_path = hi_l0_test_data_path / "H90_NHK_20241104.bin"
    processed_data = hi_l1a(packet_file_path=bin_data_path)

    assert np.unique(processed_data[0]["pkt_apid"].values) == HIAPID.H90_APP_NHK.value
    assert processed_data[0].attrs["Logical_source"] == "imap_hi_l1a_90sensor-hk"

    # Write CDF
    cem_raw_cdf_filepath = write_cdf(processed_data[0], istp=False)
    assert cem_raw_cdf_filepath.name == "imap_hi_l1a_90sensor-hk_20241105_v999.cdf"

    validation_df = pd.read_csv(hi_l0_test_data_path / "H90_NHK_20241104_verify.csv")
    for col_name, series in validation_df.items():
        # No timestamp column in dataset
        # The original binary sample data delivered was missing the last byte of
        # each packet. This was due to a configuration of the recording software.
        # Tim Plummer "fixed" the data by appending an empty byte to each packet.
        # This means that the CKSUM values will not match the validation csv.
        if col_name in ["timestamp", "CKSUM"]:
            continue
        ds_var_name = VALIDATION_TO_TEST_COLUMN_MAP.get(col_name, col_name.lower())
        np.testing.assert_array_equal(
            series.values,
            processed_data[0][ds_var_name].data,
            err_msg=f"Validation of {col_name} failed",
        )


def test_app_hist_decom(hi_l0_test_data_path):
    """Test histogram (SCI_CNT) data"""
    bin_data_path = hi_l0_test_data_path / "H90_sci_cnt_20241104.bin"
    processed_data = hi_l1a(packet_file_path=bin_data_path)

    assert processed_data[0].attrs["Logical_source"] == "imap_hi_l1a_90sensor-hist"
    # TODO: compare with validation data once we have it. Issue: #1185

    # Write CDF
    cem_raw_cdf_filepath = write_cdf(processed_data[0])

    assert cem_raw_cdf_filepath.name.startswith("imap_hi_l1a_90sensor-hist_")


def test_unpack_hist_counter():
    """Test hi.l1a.histogram.unpack_hist_counter()"""
    # To ensure correct unpacking, use expected values with ones in the upper
    # and lower parts of the 12-bit numbers
    expected = (np.arange(180).reshape((2, 90)) + 2**10).astype(">u2")
    # convert each expected uint16 to a 12-bit bitstring and join
    bin_str = "".join([f"{val:012b}" for val in expected.ravel()])
    # convert the bitstring to a bytes object
    bytes_array = int(bin_str, 2).to_bytes(len(bin_str) // 8, byteorder="big")
    output_array = unpack_hist_counter(bytes_array)
    np.testing.assert_array_equal(output_array, expected)
    assert output_array.dtype == np.uint16


def test_parse_direct_events():
    """Test coverage for parse_direct_events function."""
    # Generate fake, binary blob using random numbers
    np.random.seed(2)
    n_events = 10_000
    exp_dict = dict()
    exp_dict["trigger_id"] = np.random.randint(1, 4, size=n_events, dtype=np.uint8)
    exp_dict["de_tag"] = np.random.randint(0, 2**16, size=n_events, dtype=np.uint16)
    exp_dict["tof_1"] = np.random.randint(0, 2**10, size=n_events, dtype=np.uint16)
    exp_dict["tof_2"] = np.random.randint(0, 2**10, size=n_events, dtype=np.uint16)
    exp_dict["tof_3"] = np.random.randint(0, 2**10, size=n_events, dtype=np.uint16)

    # Encode the random events data into a bit-string
    bin_str = ""
    for i in range(n_events):
        bin_str += f"{exp_dict['de_tag'][i]:016b}"  # 16-bits for de_tag
        bin_str += f"{exp_dict['trigger_id'][i]:02b}"  # 2-bits for trigger_id
        bin_str += f"{exp_dict['tof_1'][i]:010b}"  # 10-bits for tof_1
        bin_str += f"{exp_dict['tof_2'][i]:010b}"  # 10-bits for tof_2
        bin_str += f"{exp_dict['tof_3'][i]:010b}"  # 10-bits for tof_3
    # Convert the bit-string into a bytes object
    bytes_obj = bytes([int(bin_str[i : i + 8], 2) for i in range(0, len(bin_str), 8)])
    # Parse the fake events and check values
    de_dict = parse_direct_events(bytes_obj)
    for key, expected_val in exp_dict.items():
        np.testing.assert_array_equal(de_dict[key], expected_val)


def test_create_de_dataset():
    """Test create_dataset"""
    # dummy data to test create_dataset
    n_packets = 3
    n_events = 4
    data_dict = {
        "ccsds_met": np.arange(n_packets) + 433522961,
        "src_seq_ctr": np.arange(n_packets) + 10,
        "pkt_len": np.full(n_packets, 146),
        "last_spin_num": np.arange(n_packets) + 3,
        "spin_invalids": np.zeros(n_packets),
        "esa_step": np.full(n_packets, 4),
        "meta_seconds": np.arange(n_packets) + 433522960,
        "meta_subseconds": np.full(n_packets, 500),
        "trigger_id": [1 + i % 3 for i in range(n_events)],
        "tof_1": [450] * n_events,
        "tof_2": [450] * n_events,
        "tof_3": [450] * n_events,
        "de_tag": [i for i in range(n_events)],
        "ccsds_index": [int(i / n_events * n_packets) for i in range(n_events)],
    }

    # Test for good data
    dataset = create_de_dataset(data_dict)
    assert dataset["epoch"].shape == (n_packets,)
    assert dataset["event_met"].shape == (n_events,)

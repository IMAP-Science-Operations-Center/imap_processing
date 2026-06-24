"""Tests the L1 processing for decommutated IDEX data"""

from copy import deepcopy
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from cdflib.xarray.xarray_to_cdf import ISTPError

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.idex.decode import _decode_sub_frame, read_bits, rice_decode
from imap_processing.idex.idex_l1a import (
    PacketParser,
    idex_l1a,
)
from imap_processing.idex.idex_utils import get_10_day_window_end_date
from imap_processing.spice.time import met_to_ttj2000ns
from imap_processing.tests.idex.conftest import TEST_L0_FILE_SCI
from imap_processing.utils import packet_generator

TEST_DATA_DIR = f"{imap_module_directory}/tests/idex/test_data"


def _science_dataset(parser: PacketParser) -> xr.Dataset:
    """Return the parsed L1A science dataset regardless of container layout."""
    data = parser.data
    if isinstance(data, xr.Dataset):
        return data
    if isinstance(data, dict):
        if "l1a_sci-10days" in data:
            return data["l1a_sci-10days"]
        for dataset in data.values():
            if isinstance(dataset, xr.Dataset) and "TOF_High" in dataset:
                return dataset
    if isinstance(data, (list, tuple)):
        for dataset in data:
            if isinstance(dataset, xr.Dataset) and "TOF_High" in dataset:
                return dataset
    raise AssertionError("Unable to locate science dataset in PacketParser output")


def test_idex_cdf_file(decom_test_data_sci: xr.Dataset):
    """Verify the CDF file can be created with no errors.

    Parameters
    ----------
    decom_test_data_sci : xarray.Dataset
        The dataset to test with
    """
    file_name = write_cdf(decom_test_data_sci)

    assert file_name.exists()
    assert file_name.name == "imap_idex_l1a_sci-10days_20231218_v001.0001.cdf"
    written_dataset = load_cdf(file_name)
    assert written_dataset["time_low_sample_rate"].attrs["VAR_NOTES"] == (
        "The low sample rate is 4.0625 MHz, so adjacent samples are separated "
        "by approximately 1/4.0625 microseconds."
    )


def test_bad_cdf_attributes(decom_test_data_sci: xr.Dataset):
    """Ensure an ``ISTPError`` is raised when using bad CDF attributes.

    Parameters
    ----------
    decom_test_data_sci : xarray.Dataset
        The dataset to test with
    """
    tof_catdesc = decom_test_data_sci["TOF_High"].attrs["CATDESC"]
    del decom_test_data_sci["TOF_High"].attrs["CATDESC"]

    with pytest.raises(ISTPError):
        write_cdf(decom_test_data_sci, istp=True, terminate_on_warning=True)

    # Add attributes back so future tests do not fail
    decom_test_data_sci["TOF_High"].attrs["CATDESC"] = tof_catdesc


def test_bad_cdf_file_data(decom_test_data_sci: xr.Dataset):
    """Ensure an ``ISTPError`` is raised when using bad data.

    Parameters
    ----------
    decom_test_data_sci : xarray.Dataset
        The dataset to test with
    """
    bad_data_attrs = {
        "CATDESC": "Bad_Data",
        "DEPEND_0": "epoch",
        "DISPLAY_TYPE": "no_plot",
        "FIELDNAM": "Bad_Data",
        "FILLVAL": "",
        "FORMAT": "E12.2",
        "LABLAXIS": "Bad_Data",
        "UNITS": "",
        "VALIDMIN": "1",
        "VALIDMAX": "50",
        "VAR_TYPE": "support_data",
        "VAR_NOTES": """How did this data end up in here?
                        The CDF creation better fail.""",
    }
    bad_data_xr = xr.DataArray(
        name="bad_data",
        data=np.linspace(1, 50, 50),
        dims=("bad_data"),
        attrs=bad_data_attrs,
    )
    decom_test_data_sci["Bad_data"] = bad_data_xr

    with pytest.raises(ISTPError):
        write_cdf(decom_test_data_sci, istp=True, terminate_on_warning=True)

    del decom_test_data_sci["Bad_data"]


def test_idex_l1a_decom():
    """Verify idex_l1a function returns the correct datasets."""
    with mock.patch(
        "imap_processing.idex.idex_l1a.get_10_day_window_end_date"
    ) as mock_get_window_end_date:
        mock_get_window_end_date.return_value = "20231228"
        datasets = idex_l1a([TEST_L0_FILE_SCI, TEST_L0_FILE_SCI], "20231218")

    assert len(datasets) == 2
    # We should have 14 science events
    # The duplicates will be dropped.
    assert len(datasets[0].epoch) == 14


def test_idex_l1a_decom_no_data(caplog):
    """Verify idex_l1a function returns None if there is no data for the window."""
    datasets = idex_l1a([TEST_L0_FILE_SCI], "20260101")
    # If there is no data in the window we expect an empty list
    assert datasets == []
    # We also expect a warning to be logged that no data was found for the window
    message = (
        "No data found for dates 820497669184000000 - 821275269184000000 for"
        " l1a_msg-10days"
    )
    assert message in caplog.text


def test_idex_l1a_invalid_window_start():
    """Verify that the idex_l1a function raises an error with an invalid start date."""
    with pytest.raises(
        ValueError, match="Start date 20231218 is not an IDEX defined start date"
    ):
        idex_l1a([TEST_L0_FILE_SCI], "20231218")


def test_incomplete_event(caplog):
    """Verify that a CDF is still produced if a packet is dropped.

    The IDEX team requests that a warning be logged for incomplete events
    (dropped packets) in the data, while still allowing the CDF to be created with
    the remainder of the complete events.
    """
    xml = (
        f"{imap_module_directory}/idex/packet_definitions/"
        f"idex_science_packet_definition.xml"
    )
    caplog.at_level("WARNING")
    packets = list(packet_generator(TEST_L0_FILE_SCI, xml))
    packets = packets[0:1] + packets[2:]
    with mock.patch(
        "imap_processing.idex.idex_l1a.decom_packets",
        return_value=(packets, xr.Dataset(), xr.Dataset()),
    ):
        l1a_dataset = PacketParser(TEST_L0_FILE_SCI).data["l1a_sci-10days"]
    # Assert that all the events are present except for one.
    assert len(l1a_dataset["epoch"]) == 13
    assert "Missing packet for event number 1" in caplog.text


def test_duplicate_science_fragment_is_dropped(
    decom_test_data_sci: xr.Dataset, caplog: pytest.LogCaptureFixture
):
    """Verify exact duplicate science fragments are ignored."""
    xml = (
        f"{imap_module_directory}/idex/packet_definitions/"
        f"idex_science_packet_definition.xml"
    )
    packets = list(packet_generator(TEST_L0_FILE_SCI, xml))
    duplicate_packet = deepcopy(packets[1])
    packets = [*packets[:2], duplicate_packet, *packets[2:]]

    with caplog.at_level("WARNING"):
        with mock.patch(
            "imap_processing.idex.idex_l1a.decom_packets",
            return_value=(packets, xr.Dataset(), xr.Dataset()),
        ):
            deduped = _science_dataset(PacketParser(TEST_L0_FILE_SCI))

    xr.testing.assert_equal(deduped, decom_test_data_sci)
    assert "Skipping duplicate copy" in caplog.text


def test_duplicate_header_packet_is_dropped(
    decom_test_data_sci: xr.Dataset, caplog: pytest.LogCaptureFixture
):
    """Verify exact duplicate header packets are ignored."""
    xml = (
        f"{imap_module_directory}/idex/packet_definitions/"
        f"idex_science_packet_definition.xml"
    )
    packets = list(packet_generator(TEST_L0_FILE_SCI, xml))
    duplicate_header = deepcopy(packets[0])
    packets = [packets[0], duplicate_header, *packets[1:]]

    with caplog.at_level("WARNING"):
        with mock.patch(
            "imap_processing.idex.idex_l1a.decom_packets",
            return_value=(packets, xr.Dataset(), xr.Dataset()),
        ):
            deduped = _science_dataset(PacketParser(TEST_L0_FILE_SCI))

    xr.testing.assert_equal(deduped, decom_test_data_sci)
    assert "Duplicate header packet for event" in caplog.text


def test_shorter_duplicate_science_fragment_is_replaced(
    decom_test_data_sci: xr.Dataset, caplog: pytest.LogCaptureFixture
):
    """Verify a longer retransmitted fragment replaces a shorter copy."""
    xml = (
        f"{imap_module_directory}/idex/packet_definitions/"
        f"idex_science_packet_definition.xml"
    )
    packets = list(packet_generator(TEST_L0_FILE_SCI, xml))
    shorter_packet = deepcopy(packets[1])
    shorter_packet["IDX__SCI0RAW"] = bytes(shorter_packet["IDX__SCI0RAW"])[:-10]
    packets = [*packets[:1], shorter_packet, *packets[1:]]

    with caplog.at_level("WARNING"):
        with mock.patch(
            "imap_processing.idex.idex_l1a.decom_packets",
            return_value=(packets, xr.Dataset(), xr.Dataset()),
        ):
            deduped = _science_dataset(PacketParser(TEST_L0_FILE_SCI))

    xr.testing.assert_equal(deduped, decom_test_data_sci)
    assert "Replacing shorter science fragment" in caplog.text


def test_shorter_duplicate_science_fragment_is_ignored(
    decom_test_data_sci: xr.Dataset, caplog: pytest.LogCaptureFixture
):
    """Verify a shorter retransmitted fragment is ignored."""
    xml = (
        f"{imap_module_directory}/idex/packet_definitions/"
        f"idex_science_packet_definition.xml"
    )
    packets = list(packet_generator(TEST_L0_FILE_SCI, xml))
    shorter_packet = deepcopy(packets[1])
    shorter_packet["IDX__SCI0RAW"] = bytes(shorter_packet["IDX__SCI0RAW"])[:-10]
    packets = [*packets[:2], shorter_packet, *packets[2:]]

    with caplog.at_level("WARNING"):
        with mock.patch(
            "imap_processing.idex.idex_l1a.decom_packets",
            return_value=(packets, xr.Dataset(), xr.Dataset()),
        ):
            deduped = _science_dataset(PacketParser(TEST_L0_FILE_SCI))

    xr.testing.assert_equal(deduped, decom_test_data_sci)
    assert "Ignoring shorter duplicate science fragment" in caplog.text


def test_conflicting_duplicate_science_fragment_skips_event(
    decom_test_data_sci: xr.Dataset, caplog: pytest.LogCaptureFixture
):
    """Verify conflicting duplicate fragments cause the event to be skipped."""
    xml = (
        f"{imap_module_directory}/idex/packet_definitions/"
        f"idex_science_packet_definition.xml"
    )
    packets = list(packet_generator(TEST_L0_FILE_SCI, xml))
    conflicting_packet = deepcopy(packets[1])
    raw = bytearray(bytes(conflicting_packet["IDX__SCI0RAW"]))
    raw[-1] ^= 0x01
    conflicting_packet["IDX__SCI0RAW"] = bytes(raw)
    packets = [*packets[:2], conflicting_packet, *packets[2:]]

    with caplog.at_level("WARNING"):
        with mock.patch(
            "imap_processing.idex.idex_l1a.decom_packets",
            return_value=(packets, xr.Dataset(), xr.Dataset()),
        ):
            deduped = _science_dataset(PacketParser(TEST_L0_FILE_SCI))

    assert len(deduped["epoch"]) == len(decom_test_data_sci["epoch"]) - 1
    assert "Conflicting duplicate packet for event number" in caplog.text


def test_idex_tof_high_data_from_cdf(decom_test_data_sci: xr.Dataset):
    """Verify that a sample of the data is correct inside the CDF file.

    ``impact_14_tof_high_data.txt`` has been verified correct by the IDEX team

    Parameters
    ----------
    decom_test_data_sci : xarray.Dataset
        The dataset to test with
    """
    with open(
        f"{imap_module_directory}/tests/idex/test_data/impact_14_tof_high_data.txt"
    ) as f:
        data = np.array([int(line.rstrip()) for line in f])

    file_name = write_cdf(decom_test_data_sci)
    l1_data = load_cdf(file_name)
    assert (l1_data["TOF_High"][13].data == data).all()


@pytest.mark.external_test_data
def test_validate_l1a_idex_data_variables(
    decom_test_data_sci: xr.Dataset, l1a_example_data: xr.Dataset
):
    """
    Verify that each of the 6 waveform and telemetry arrays are equal to the
    corresponding array produced by the IDEX team using the same l0 file.


    Parameters
    ----------
    decom_test_data_sci : xarray.Dataset
        The dataset to test with
    l1a_example_data: xarray.Dataset
        A dataset containing the 6 waveform and telemetry arrays
    """
    # Lookup table to match the SDS array names to the Idex Team array names
    match_variables = {
        "TOF L": "TOF_Low",
        "TOF H": "TOF_High",
        "TOF M": "TOF_Mid",
        "Target H": "Target_High",
        "Target L": "Target_Low",
        "Ion Grid": "Ion_Grid",
        "Time (high sampling)": "time_high_sample_rate",
        "Time (low sampling)": "time_low_sample_rate",
        "idx__txhdrfswaidcopy": "aid",
    }
    # The Engineering data is converting to UTC, and the SDC is converting to J2000,
    # for 'epoch' and 'Timestamp' so this test is using the raw time value 'SCHOARSE' to
    # validate time
    # TODO remove the low and high time from this list after the IDEX team produces a
    #  new l1a h5 file.
    arrays_to_skip = [
        "Timestamp",
        "Epoch",
        "event",
        "Time (high sampling)",
        "Time (low sampling)",
        "IDX__SCI0AID",  # This is dropped because it is invalid
        "IDX__TXHDRFSWAIDCOPY",  # this is renamed to aid
    ]

    # loop through all keys from the l1a example dict
    for var in l1a_example_data.variables:
        if var not in arrays_to_skip:
            # Find the corresponding array name
            cdf_var = match_variables.get(var, var.lower())

            np.testing.assert_array_equal(
                decom_test_data_sci[cdf_var],
                l1a_example_data[var],
                f"The array '{cdf_var}' does not equal the expected example "
                f"array '{var}' produced by the IDEX team",
            )


def test_compressed_packet():
    """
    Test compressed data decompression against known non-compressed data.
    """

    compressed = Path(f"{TEST_DATA_DIR}/compressed_2023_102_14_24_55.pkts")
    non_compressed = Path(f"{TEST_DATA_DIR}/non_compressed_2023_102_14_22_26.pkts")

    decompressed = PacketParser(compressed).data["l1a_sci-10days"]
    expected = PacketParser(non_compressed).data["l1a_sci-10days"]

    waveforms = [
        "TOF_High",
        "TOF_Low",
        "TOF_Mid",
        "Ion_Grid",
        "Target_High",
        "Target_Low",
    ]

    # Compare each decompressed waveform with known non-compressed waveform.
    for var in waveforms:
        assert np.allclose(decompressed[var], expected[var]), (
            f"Variable: {var} is different for the decompressed and non compressed "
            f"datasets."
        )


def test_read_bits():
    """
    Test that read_bits() properly increments the pointer and reads the correct ints.
    """

    binary_str_6 = "110"
    binary_str_neg_6 = "1010"
    binary_str_1 = "0001"

    full_string = binary_str_6 + binary_str_neg_6 + binary_str_1

    pointer = 0
    val, pointer = read_bits(full_string, pointer, 3)
    assert val == 6
    assert pointer == 3
    # Test with a signed binary string
    val, pointer = read_bits(full_string, pointer, 4, signed=True)
    assert val == -6
    assert pointer == 7

    val, pointer = read_bits(full_string, pointer, 4)
    assert val == 1
    assert pointer == 11


@mock.patch("imap_processing.idex.decode._decode_sub_frame")
def test_rice_decode(mock_decode_sub_frame):
    """
    Verify that rice_decode() returns the expected list of integers.
    """

    mock_decode_sub_frame.return_value = [1 for _ in range(64)], 1
    # mock binary string.
    bstr = "0100000"
    ints = rice_decode(bstr, True, 640)
    # If there are 64 samples per block and the sample count is 640,
    # There should be 10 calls to _decode_sub_frame()
    assert mock_decode_sub_frame.call_count == 10
    # There should be 640 samples
    assert ints == [1 for _ in range(640)]


def test_decode_sub_frame_psel_0():
    """Test constant predictor (psel=0) where all samples equal first sample."""
    psel = 0
    bstring = "0" * 9 + "1"  # 1 in 10-bit binary
    ints, bp = _decode_sub_frame(bstring, bp=0, psel=psel, k=0, n_bits=10)
    assert ints == [1 for _ in range(64)]


def test_decode_sub_frame_psel_1():
    """Test verbatim predictor (psel=1) where each sample stored directly."""
    # Create a binary string that is equal to 1, 2, 3, 4 to 64. Each int is represented
    # in 10 bits to mock the expected high sample waveform data.
    psel = 1
    bstring = "".join([bin(i)[2:].zfill(10) for i in range(64)])
    ints, bp = _decode_sub_frame(bstring, bp=0, psel=psel, k=0, n_bits=10)
    assert ints == [i for i in range(64)]


def test_decode_sub_frame_psel_2():
    """Test linear predictor 1 (psel=2) with rice encoded residuals."""
    psel = 2
    k = 1
    # This encoding takes the signed residual value (current value - last value)
    # Example values 1, 2, 3, 8
    # residual of 1
    # quotient = 1 >> (k+1) = 0 (unary = "1")
    # remainder = "01"
    residual_1_and_2 = "101"
    # residual of 5
    # quotient = 4 >> (k+1) = 1 maps to 2 (unary = "001")
    # remainder = "01"
    residual_3 = "00101"
    warmup = "0" * 9 + "1"  # equal to 1

    bstring = warmup + residual_1_and_2 + residual_1_and_2 + residual_3
    ints, bp = _decode_sub_frame(bstring, bp=0, psel=psel, k=k, n_bits=10)
    assert ints == [1, 2, 3, 8]


def test_decode_sub_frame_psel_3():
    """Test linear predictor 2 (psel=3) with rice encoded residuals."""
    # This predictor assumes that sample X(n) = 2X(n-1)-X(n-2). This predictor
    # calculates the slope of the signal based on the two previous values
    # (slope = X(n-1)-X(n-2)) and predicts the value of X(n) will follow on a straight
    # line drawn between the two previous points.
    # X(n) = X(n-1) + (X(n-1)-X(n-2))  = 2X(n-1)-X(n-2).
    psel = 3
    k = 1
    # Example values: [1, 2, 4, 1, 5]
    # For X(4)=1: predicted=3 (2*2-1), actual=4, residual=1
    # For X(1)=-5: predicted=6 (2*4-2), actual=1, residual=-5
    # For X(5)=7: predicted=-2 (2*1-4), actual=5, residual=7
    # residual of 1
    # quotient = 1 >> (K+1) = 0 (unary = "1")
    # remainder = "01"
    residual_1 = "101"
    # residual of -5
    # quotient = -5 >> (K+1) = -2 maps to 3 (unary = "0001")
    # remainder = "11"
    residual_2 = "000111"
    # residual of 7
    # quotient = 7 >> (K+1) = 1 maps to 2 (unary = "001")
    # remainder = "11"
    residual_3 = "00111"
    warmup1 = "0" * 9 + "1"  # 1
    warmup2 = "0" * 8 + "10"  # 2

    bstring = warmup1 + warmup2 + residual_1 + residual_2 + residual_3
    ints, bp = _decode_sub_frame(bstring, bp=0, psel=psel, k=k, n_bits=10)
    assert ints == [1, 2, 4, 1, 5]


def test_catlst_dataset(decom_test_data_catlst: list[xr.Dataset]):
    """Verify that the dataset contains what we expect and can be written to a cdf.

    Parameters
    ----------
    decom_test_data_catlst : list[xarray.Dataset]
        The raw and derived (l1a and l1b) datasets to test with.
    """
    for ds in decom_test_data_catlst:
        assert "shcoarse" in ds
        assert "shfine" in ds
        # Assert epoch is calculated using fine-grained clock ticks
        expected_epoch = met_to_ttj2000ns(ds["shcoarse"] + ds["shfine"] * 20e-6)
        np.testing.assert_array_equal(ds.epoch, expected_epoch)
    # Assert that the dataset can be written to a CDF file
    filename_l1a = write_cdf(decom_test_data_catlst[0])
    assert filename_l1a.name == "imap_idex_l1a_catlst-10days_20241206_v001.0001.cdf"

    filename_l1b = write_cdf(decom_test_data_catlst[1])
    assert filename_l1b.name == "imap_idex_l1b_catlst-10days_20241206_v001.0001.cdf"


def test_msg_dataset(decom_test_data_msg: xr.Dataset):
    """Verify that the dataset contains what we expect and can be written to a cdf.

    Parameters
    ----------
    decom_test_data_msg : xarray.Dataset
        The raw l1a dataset to test with.
    """
    assert "elsec_evtpkt" in decom_test_data_msg
    assert "elssec_evtpkt" in decom_test_data_msg
    # Assert epoch is calculated using fine grained clock ticks
    expected_epoch = met_to_ttj2000ns(
        decom_test_data_msg["elsec_evtpkt"]
        + decom_test_data_msg["elssec_evtpkt"] * 20e-6
    )
    np.testing.assert_array_equal(decom_test_data_msg.epoch, expected_epoch)
    # Assert that the dataset can be written to a CDF file
    filename_l1a = write_cdf(decom_test_data_msg)
    assert filename_l1a.name == "imap_idex_l1a_msg-10days_20100101_v001.0001.cdf"

    # Validate the messages with the IDEX team example data
    example_data = pd.read_csv(
        f"{TEST_DATA_DIR}/idex_event_messages.csv", skiprows=1, header=None
    )

    messages = example_data.iloc[:, 1].tolist()
    np.testing.assert_array_equal(decom_test_data_msg["messages"].data, messages)


def test_get_window_end_date():
    """Verify that the end date is returned for a 10-day window."""
    assert get_10_day_window_end_date("20260101") == "20260110"
    assert get_10_day_window_end_date("20261226") == "20270101"

    with pytest.raises(
        ValueError,
        match="Start date 20260102 is not an IDEX defined "
        "start date for a 10-day window.",
    ):
        # This invalid start date should raise an error.
        get_10_day_window_end_date("20260102")


def test_get_window_invalid_lookup():
    """Verify that an invalid lookup table raises an error."""
    with patch(
        "imap_processing.idex.idex_utils.IDEX_10_DAY_RANGES_PATH",
        "imap_processing/tests/idex/test_data/test_idex_10_day_window.csv",
    ):
        message = (
            "There should only be one row where start_date is equal to 20250101. "
            "Please check lookup table"
        )
        with pytest.raises(ValueError, match=message):
            get_10_day_window_end_date("20250101")

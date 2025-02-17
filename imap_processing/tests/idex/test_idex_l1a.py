"""Tests the L1 processing for decommutated IDEX data"""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import xarray as xr
from cdflib.xarray.xarray_to_cdf import ISTPError

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.idex.decode import _decode_sub_frame, read_bits, rice_decode
from imap_processing.idex.idex_l1a import PacketParser


def test_idex_cdf_file(decom_test_data: xr.Dataset):
    """Verify the CDF file can be created with no errors.

    Parameters
    ----------
    decom_test_data : xarray.Dataset
        The dataset to test with
    """

    file_name = write_cdf(decom_test_data)

    assert file_name.exists()
    assert file_name.name == "imap_idex_l1a_sci_20231214_v001.cdf"


def test_bad_cdf_attributes(decom_test_data: xr.Dataset):
    """Ensure an ``ISTPError`` is raised when using bad CDF attributes.

    Parameters
    ----------
    decom_test_data : xarray.Dataset
        The dataset to test with
    """
    tof_catdesc = decom_test_data["TOF_High"].attrs["CATDESC"]
    del decom_test_data["TOF_High"].attrs["CATDESC"]

    with pytest.raises(ISTPError):
        write_cdf(decom_test_data)

    # Add attributes back so future tests do not fail
    decom_test_data["TOF_High"].attrs["CATDESC"] = tof_catdesc


def test_bad_cdf_file_data(decom_test_data: xr.Dataset):
    """Ensure an ``ISTPError`` is raised when using bad data.

    Parameters
    ----------
    decom_test_data : xarray.Dataset
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
    decom_test_data["Bad_data"] = bad_data_xr

    with pytest.raises(ISTPError):
        write_cdf(decom_test_data)

    del decom_test_data["Bad_data"]


def test_idex_tof_high_data_from_cdf(decom_test_data: xr.Dataset):
    """Verify that a sample of the data is correct inside the CDF file.

    ``impact_14_tof_high_data.txt`` has been verified correct by the IDEX team

    Parameters
    ----------
    decom_test_data : xarray.Dataset
        The dataset to test with
    """
    with open(
        f"{imap_module_directory}/tests/idex/test_data/impact_14_tof_high_data.txt"
    ) as f:
        data = np.array([int(line.rstrip()) for line in f])

    file_name = write_cdf(decom_test_data)
    l1_data = load_cdf(file_name)
    assert (l1_data["TOF_High"][13].data == data).all()


def test_compressed_packet():
    """
    Test compressed data decompression against known non-compressed data.
    """
    test_data_dir = f"{imap_module_directory}/tests/idex/test_data"

    compressed = Path(f"{test_data_dir}/compressed_2023_102_14_24_55.pkts")
    non_compressed = Path(f"{test_data_dir}/non_compressed_2023_102_14_22_26.pkts")

    decompressed = PacketParser(compressed, "001").data
    expected = PacketParser(non_compressed, "001").data

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

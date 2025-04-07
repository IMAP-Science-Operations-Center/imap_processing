"""Tests coverage for imap_processing.cli."""

import sys
from unittest import mock

import pytest
import xarray as xr
from imap_data_access.processing_input import (
    ProcessingInputCollection,
    ScienceInput,
)

from imap_processing.cli import (
    Codice,
    Hi,
    Hit,
    Spacecraft,
    Swe,
    Ultra,
    _validate_args,
    main,
)


@pytest.fixture
def mock_instrument_dependencies():
    with (
        mock.patch("imap_processing.cli.imap_data_access.query") as mock_query,
        mock.patch("imap_processing.cli.imap_data_access.download") as mock_download,
        mock.patch("imap_processing.cli.imap_data_access.upload") as mock_upload,
        mock.patch("imap_processing.cli.write_cdf") as mock_write_cdf,
        mock.patch("imap_processing.cli.load_cdf") as mock_load_cdf,
        mock.patch(
            "imap_processing.cli.ProcessInstrument.pre_processing"
        ) as mock_pre_processing,
    ):
        mocks = {
            "mock_query": mock_query,
            "mock_download": mock_download,
            "mock_upload": mock_upload,
            "mock_write_cdf": mock_write_cdf,
            "mock_load_cdf": mock_load_cdf,
            "mock_pre_processing": mock_pre_processing,
        }
        yield mocks


@mock.patch("imap_processing.cli.Mag")
def test_main(mock_instrument):
    """Test imap_processing.cli.main()"""
    test_args = [
        "imap_cli",
        "--instrument",
        "mag",
        "--dependency",
        (
            "["
            "{"
            '"type": "ancillary",'
            '"files": ['
            '"imap_mag_l1b-cal_20250101_v001.cdf",'
            '"imap_mag_l1b-cal_20250103-20250104_v002.cdf"'
            "]"
            "},"
            "{"
            '"type": "science",'
            '"files": ['
            '"imap_mag_l0_raw_20240430_v001.cdf",'
            "]"
            "}"
            "]"
        ),
        "--data-level",
        "l1a",
        "--start-date",
        "20240430",
        "--end-date",
        "20240501",
        "--version",
        "v001",
        "--upload-to-sdc",
    ]
    with mock.patch.object(sys, "argv", test_args):
        # Running without raising an exception is a pass.
        # No asserts needed.
        main()


@pytest.mark.parametrize(
    "instrument, data_level, raises_value_error",
    [
        ("mag", "l1a", ""),
        ("foo", "l1a", "foo is not in the supported .*"),
        ("codice", "l1z", "l1z is not a supported .*"),
    ],
)
def test_validate_args(instrument, data_level, raises_value_error):
    """Test coverage for imap_processing.cli._validate_args()"""
    args = mock.Mock
    args.instrument = instrument
    args.data_level = data_level

    if raises_value_error:
        with pytest.raises(ValueError, match=raises_value_error):
            _validate_args(args)
    else:
        _validate_args(args)


@mock.patch("imap_processing.cli.codice_l1a.process_codice_l1a")
def test_codice(mock_codice_l1a, mock_instrument_dependencies):
    """Test coverage for cli.CoDICE class"""

    test_dataset = xr.Dataset({}, attrs={"cdf_filename": "file0"})
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_codice_l0_raw_20230822_v001.pkts")
    )
    mocks = mock_instrument_dependencies
    mocks["mock_query"].return_value = [{"file_path": "/path/to/file0"}]
    mocks["mock_download"].return_value = "file0"
    mock_codice_l1a.return_value = [test_dataset]
    mocks["mock_write_cdf"].side_effect = ["/path/to/file0"]
    mocks["mock_pre_processing"].return_value = input_collection

    dependency_str = (
        '[{"type": "science","files": ["imap_codice_l0_raw_20230822_v001.pkts"]}]'
    )

    instrument = Codice(
        "l1a", "hskp", dependency_str, "20230822", "20230822", "v001", True
    )

    instrument.process()
    assert mock_codice_l1a.call_count == 1
    assert mocks["mock_upload"].call_count == 1


@pytest.mark.parametrize(
    "data_level, science_input, n_prods",
    [
        ("l1a", ["imap_hi_l0_raw_20231212_v001.pkts"], 2),
        ("l1b", ["imap_hi_l1a_90sensor-de_20241105_v001.cdf"], 1),
        ("l1b", ["imap_hi_l0_raw_20231212_v001.pkts"], 2),
        ("l1c", ["imap_hi_l1b_45sensor-de_20250415_v001.cdf"], 1),
    ],
)
def test_hi_l1(mock_instrument_dependencies, data_level, science_input, n_prods):
    """Test coverage for cli.Hi class"""
    mocks = mock_instrument_dependencies
    mocks["mock_write_cdf"].side_effect = ["/path/to/file0"] * n_prods
    mocks["mock_load_cdf"].return_value = xr.Dataset()
    input_collection = ProcessingInputCollection(
        *[ScienceInput(file) for file in science_input]
    )
    mocks["mock_pre_processing"].return_value = input_collection

    # patch autospec=True makes this test confirm that the function call in cli.py
    # matches the mocked function signature.
    with mock.patch(
        f"imap_processing.cli.hi_{data_level}.hi_{data_level}", autospec=True
    ) as mock_hi:
        mock_hi.return_value = [f"{data_level}_file{n}" for n in range(n_prods)]
        dependency_str = (
            '[{"type": "science","files": ["imap_hi_l0_raw_20231212_v001.pkts"]}]'
        )
        instrument = Hi(
            data_level, "sci", dependency_str, "20231212", "20231213", "v005", True
        )

        instrument.process()
        assert mock_hi.call_count == 1
        assert mocks["mock_upload"].call_count == n_prods


@mock.patch("imap_processing.cli.quaternions.process_quaternions", autospec=True)
def test_spacecraft(mock_spacecraft_l1a, mock_instrument_dependencies):
    """Test coverage for cli.Spacecraft class"""

    test_dataset = xr.Dataset({}, attrs={"cdf_filename": "file0"})
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_spacecraft_l0_raw_20230822_v001.pkts")
    )
    mocks = mock_instrument_dependencies
    mocks["mock_query"].return_value = [{"file_path": "/path/to/file0"}]
    mocks["mock_download"].return_value = "file0"
    mock_spacecraft_l1a.return_value = [test_dataset]
    mocks["mock_write_cdf"].side_effect = ["/path/to/file0"]
    mocks["mock_pre_processing"].return_value = input_collection

    dependency_str = (
        '[{"type": "science","files": ["imap_spacecraft_l0_raw_20230822_v001.pkts"]}]'
    )

    instrument = Spacecraft(
        "l1a", "quaternions", dependency_str, "20230822", "20230822", "v001", True
    )

    instrument.process()
    assert mock_spacecraft_l1a.call_count == 1
    assert mocks["mock_upload"].call_count == 1


@mock.patch("imap_processing.cli.ultra_l1a.ultra_l1a")
def test_ultra_l1a(mock_ultra_l1a, mock_instrument_dependencies):
    """Test coverage for cli.Ultra class with l1a data level"""
    mocks = mock_instrument_dependencies
    mock_ultra_l1a.return_value = ["l1a_dataset0", "l1a_dataset1"]
    mocks["mock_write_cdf"].side_effect = ["/path/to/product0", "/path/to/product1"]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_ultra_l0_raw_20240207_v001.pkts")
    )
    mocks["mock_pre_processing"].return_value = input_collection

    dependency_str = (
        '[{"type": "science","files": ["imap_ultra_l0_raw_20240207_v001.pkts"]}]'
    )
    instrument = Ultra(
        "l1a", "raw", dependency_str, "20240207", "20240208", "v001", True
    )

    instrument.process()
    assert mock_ultra_l1a.call_count == 1
    assert mocks["mock_upload"].call_count == 2


@mock.patch("imap_processing.cli.ultra_l1b.ultra_l1b")
def test_ultra_l1b(mock_ultra_l1b, mock_instrument_dependencies):
    """Test coverage for cli.Ultra class with l1b data level"""
    mocks = mock_instrument_dependencies
    mocks["mock_download"].return_value = "dependency0"
    mock_ultra_l1b.return_value = ["l1b_dataset0", "l1b_dataset1"]
    mocks["mock_write_cdf"].side_effect = ["/path/to/product0", "/path/to/product1"]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_ultra_l1a_de_20240207_v001.cdf")
    )
    mocks["mock_pre_processing"].return_value = input_collection

    instrument = Ultra("l1b", "de", "[]", "20240207", "20240208", "v001", True)

    instrument.process()
    assert mocks["mock_download"].call_count == 0
    assert mock_ultra_l1b.call_count == 1
    assert mocks["mock_upload"].call_count == 2


@mock.patch("imap_processing.cli.ultra_l1c.ultra_l1c")
def test_ultra_l1c(mock_ultra_l1c, mock_instrument_dependencies):
    """Test coverage for cli.Ultra class with l1c data level"""
    mocks = mock_instrument_dependencies
    mock_ultra_l1c.return_value = ["l1c_dataset0", "l1c_dataset1"]
    mocks["mock_write_cdf"].side_effect = ["/path/to/product0", "/path/to/product1"]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_ultra_l1b_de_20240207_v001.cdf")
    )
    mocks["mock_pre_processing"].return_value = input_collection

    instrument = Ultra("l1c", "pset", "[]", "20240207", "20240208", "v001", True)

    instrument.process()
    assert mock_ultra_l1c.call_count == 1
    assert mocks["mock_upload"].call_count == 2


@mock.patch("imap_processing.cli.hit_l1a")
def test_hit_l1a(mock_hit_l1a, mock_instrument_dependencies):
    """Test coverage for cli.Hit class with l1a data level"""
    mocks = mock_instrument_dependencies
    mock_hit_l1a.return_value = ["l1a_dataset0", "l1a_dataset1"]
    mocks["mock_write_cdf"].side_effect = ["/path/to/product0", "/path/to/product1"]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_hit_l0_raw_20100105_v001.pkts")
    )
    mocks["mock_pre_processing"].return_value = input_collection

    dependency_str = (
        '[{"type": "science","files": ["imap_hit_l0_raw_20100105_v001.pkts"]}]'
    )
    instrument = Hit("l1a", "raw", dependency_str, "20100105", "20100101", "v001", True)

    instrument.process()
    assert mock_hit_l1a.call_count == 1
    assert mocks["mock_upload"].call_count == 2


@mock.patch("imap_processing.cli.swe_l1a")
def test_post_processing(mock_swe_l1a, mock_instrument_dependencies):
    """Test coverage for post processing"""
    mocks = mock_instrument_dependencies
    mocks["mock_download"].return_value = "dependency0"
    # Return empty list to simulate no data to write
    mock_swe_l1a.return_value = []
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_swe_l0_raw_20100105_v001.pkts")
    )
    mocks["mock_pre_processing"].return_value = input_collection

    dependency_str = (
        '[{"type": "science","files": ["imap_swe_l0_raw_20100105_v001.pkts"]}]'
    )
    instrument = Swe("l1a", "raw", dependency_str, "20100105", "20100101", "v001", True)

    # This function calls both the instrument.do_processing() and
    # instrument.post_processing()
    instrument.process()
    assert mock_swe_l1a.call_count == 1
    # This test is testing that no upload happened
    assert mocks["mock_upload"].call_count == 0

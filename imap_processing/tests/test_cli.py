"""Tests coverage for imap_processing.cli."""

import sys
from unittest import mock

import pytest
import xarray as xr

from imap_processing.cli import Codice, Hi, Hit, Ultra, _validate_args, main


@pytest.fixture()
def mock_instrument_dependencies():
    with (
        mock.patch("imap_processing.cli.imap_data_access.query") as mock_query,
        mock.patch("imap_processing.cli.imap_data_access.download") as mock_download,
        mock.patch("imap_processing.cli.imap_data_access.upload") as mock_upload,
        mock.patch("imap_processing.cli.write_cdf") as mock_write_cdf,
        mock.patch("imap_processing.cli.load_cdf") as mock_load_cdf,
    ):
        mocks = {
            "mock_query": mock_query,
            "mock_download": mock_download,
            "mock_upload": mock_upload,
            "mock_write_cdf": mock_write_cdf,
            "mock_load_cdf": mock_load_cdf,
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
            '[{"instrument": "mag", '
            '"data_level": "l0", '
            '"descriptor": "sci", '
            '"version": "v001", '
            '"start_date": "20240430"}]'
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
        ("mag", "l0", ""),
        ("foo", "l0", "foo is not in the supported .*"),
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

    mocks = mock_instrument_dependencies
    mocks["mock_query"].return_value = [{"file_path": "/path/to/file0"}]
    mocks["mock_download"].return_value = "file0"
    mock_codice_l1a.return_value = test_dataset
    mocks["mock_write_cdf"].side_effect = ["/path/to/file0"]

    dependency_str = (
        "[{"
        "'instrument': 'codice',"
        "'data_level': 'l0',"
        "'descriptor': 'hskp',"
        "'version': 'v001',"
        "'start_date': '20230822'"
        "}]"
    )
    instrument = Codice("l1a", dependency_str, "20230822", "20230822", "v001", True)
    instrument.process()
    assert mocks["mock_query"].call_count == 1
    assert mocks["mock_download"].call_count == 1
    assert mock_codice_l1a.call_count == 1
    assert mocks["mock_upload"].call_count == 1


@pytest.mark.parametrize("data_level, n_prods", [("l1a", 2), ("l1b", 1), ("l1c", 1)])
def test_hi_l1(mock_instrument_dependencies, data_level, n_prods):
    """Test coverage for cli.Hi class"""
    mocks = mock_instrument_dependencies
    mocks["mock_query"].return_value = [{"file_path": "/path/to/file0"}]
    mocks["mock_download"].return_value = "file0"
    mocks["mock_write_cdf"].side_effect = ["/path/to/file0", "/path/to/file1"]
    mocks["mock_load_cdf"].return_value = xr.Dataset()

    # patch autospec=True makes this test confirm that the function call in cli.py
    # matches the mocked function signature.
    with mock.patch(
        f"imap_processing.cli.hi_{data_level}.hi_{data_level}", autospec=True
    ) as mock_hi:
        mock_hi.return_value = [f"{data_level}_file{n}" for n in range(n_prods)]
        dependency_str = (
            "[{"
            "'instrument': 'lo',"
            "'data_level': 'l0',"
            "'descriptor': 'sci',"
            "'version': 'v00-01',"
            "'start_date': '20231212'"
            "}]"
        )
        instrument = Hi(
            data_level, dependency_str, "20231212", "20231213", "v005", True
        )
        instrument.process()
        assert mocks["mock_query"].call_count == 1
        assert mocks["mock_download"].call_count == 1
        assert mock_hi.call_count == 1
        assert mocks["mock_upload"].call_count == n_prods


@mock.patch("imap_processing.cli.ultra_l1a.ultra_l1a")
def test_ultra_l1a(mock_ultra_l1a, mock_instrument_dependencies):
    """Test coverage for cli.Ultra class with l1a data level"""
    mocks = mock_instrument_dependencies
    mocks["mock_query"].return_value = [{"file_path": "/path/to/file0"}]
    mocks["mock_download"].return_value = "dependency0"
    mock_ultra_l1a.return_value = ["l1a_dataset0", "l1a_dataset1"]
    mocks["mock_write_cdf"].side_effect = ["/path/to/product0", "/path/to/product1"]

    dependency_str = (
        "[{"
        "'instrument': 'ultra',"
        "'data_level': 'l0',"
        "'descriptor': 'raw',"
        "'version': 'v001',"
        "'start_date': '20240207'"
        "}]"
    )
    instrument = Ultra("l1a", dependency_str, "20240207", "20240208", "v001", True)
    instrument.process()
    assert mocks["mock_query"].call_count == 1
    assert mocks["mock_download"].call_count == 1
    assert mock_ultra_l1a.call_count == 1
    assert mocks["mock_upload"].call_count == 2


@mock.patch("imap_processing.cli.ultra_l1b.ultra_l1b")
def test_ultra_l1b(mock_ultra_l1b, mock_instrument_dependencies):
    """Test coverage for cli.Ultra class with l1b data level"""
    mocks = mock_instrument_dependencies
    mocks["mock_query"].return_value = [{"file_path": "/path/to/file0"}]
    mocks["mock_download"].return_value = "dependency0"
    mock_ultra_l1b.return_value = ["l1b_dataset0", "l1b_dataset1"]
    mocks["mock_write_cdf"].side_effect = ["/path/to/product0", "/path/to/product1"]

    instrument = Ultra("l1b", "[]", "20240207", "20240208", "v001", True)
    instrument.process()
    assert mocks["mock_query"].call_count == 0
    assert mocks["mock_download"].call_count == 0
    assert mock_ultra_l1b.call_count == 1
    assert mocks["mock_upload"].call_count == 2


@mock.patch("imap_processing.cli.ultra_l1c.ultra_l1c")
def test_ultra_l1c(mock_ultra_l1c, mock_instrument_dependencies):
    """Test coverage for cli.Ultra class with l1c data level"""
    mocks = mock_instrument_dependencies
    mocks["mock_query"].return_value = [{"file_path": "/path/to/file0"}]
    mocks["mock_download"].return_value = "dependency0"
    mock_ultra_l1c.return_value = ["l1c_dataset0", "l1c_dataset1"]
    mocks["mock_write_cdf"].side_effect = ["/path/to/product0", "/path/to/product1"]

    instrument = Ultra("l1c", "[]", "20240207", "20240208", "v001", True)
    instrument.process()
    assert mocks["mock_query"].call_count == 0
    assert mocks["mock_download"].call_count == 0
    assert mock_ultra_l1c.call_count == 1
    assert mocks["mock_upload"].call_count == 2


@mock.patch("imap_processing.cli.hit_l1a")
def test_hit_l1a(mock_hit_l1a, mock_instrument_dependencies):
    """Test coverage for cli.Hit class with l1a data level"""
    mocks = mock_instrument_dependencies
    mocks["mock_query"].return_value = [{"file_path": "/path/to/file0"}]
    mocks["mock_download"].return_value = "dependency0"
    mock_hit_l1a.return_value = ["l1a_dataset0", "l1a_dataset1"]
    mocks["mock_write_cdf"].side_effect = ["/path/to/product0", "/path/to/product1"]

    dependency_str = (
        "[{"
        "'instrument': 'hit',"
        "'data_level': 'l0',"
        "'descriptor': 'raw',"
        "'version': 'v001',"
        "'start_date': '20100105'"
        "}]"
    )
    instrument = Hit("l1a", dependency_str, "20100105", "20100101", "v001", True)
    instrument.process()
    assert mocks["mock_query"].call_count == 1
    assert mocks["mock_download"].call_count == 1
    assert mock_hit_l1a.call_count == 1
    assert mocks["mock_upload"].call_count == 2

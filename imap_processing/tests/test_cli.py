"""Tests coverage for imap_processing.cli."""

import sys
from unittest import mock

import pytest

from imap_processing.cli import Hi, _validate_args, main


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
        "v00-01",
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


@mock.patch("imap_processing.cli.imap_data_access.query")
@mock.patch("imap_processing.cli.imap_data_access.download")
@mock.patch("imap_processing.cli.imap_data_access.upload")
@mock.patch("imap_processing.cli.hi_l1a.hi_l1a")
@mock.patch("imap_processing.cli.write_cdf")
def test_hi(mock_write_cdf, mock_hi_l1a, mock_upload, mock_download, mock_query):
    """Test coverage for cli.Hi class"""
    mock_query.return_value = [{"file_path": "/path/to/file0"}]
    mock_download.return_value = "file0"
    mock_hi_l1a.return_value = ["l1a_file0", "l1a_file1"]
    mock_write_cdf.side_effect = ["/path/to/file0", "/path/to/file1"]

    dependency_str = (
        "[{"
        "'instrument': 'lo',"
        "'data_level': 'l0',"
        "'descriptor': 'sci',"
        "'version': 'v00-01',"
        "'start_date': '20231212'"
        "}]"
    )
    instrument = Hi("l1a", dependency_str, "20231212", "20231213", "v005", True)
    instrument.process()
    assert mock_query.call_count == 1
    assert mock_download.call_count == 1
    assert mock_hi_l1a.call_count == 1
    assert mock_upload.call_count == 2

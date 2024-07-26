"""Testing for xtce generator template."""

import sys
from pathlib import Path
from unittest import mock

import pytest

from imap_processing.ccsds import excel_to_xtce

pytest.importorskip("openpyxl")


@pytest.fixture()
def filepath(tmpdir):
    p = Path(tmpdir / "test_file.xlsx").resolve()
    p.touch()
    return p


# General test
@mock.patch("imap_processing.ccsds.excel_to_xtce.XTCEGenerator")
def test_main_general(mock_input, filepath):
    """Testing base main function."""
    test_args = [
        "test_script",
        "--output",
        "swe.xml",
        f"{filepath}",
    ]
    with mock.patch.object(sys, "argv", test_args):
        excel_to_xtce.main()
        mock_input.assert_called_once()


# Testing without required arguments
def test_main_inval_arg():
    """Testing with invalid instrument."""
    test_args = [
        "test_script",
        "--output",
        "glows.xml",
    ]
    with mock.patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit):
            excel_to_xtce.main()


# File does not exist
def test_main_inval_file():
    """Testing with invalid file."""
    test_args = [
        "test_script",
        "--instrument",
        "glows",
        "not-a-valid-file.txt",
    ]
    with mock.patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit):
            excel_to_xtce.main()

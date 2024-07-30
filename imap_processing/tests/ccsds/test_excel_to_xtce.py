"""Testing for xtce generator template."""

import sys
from pathlib import Path
from unittest import mock

import pytest

from imap_processing.ccsds import excel_to_xtce

pytest.importorskip("openpyxl")


@pytest.fixture()
def excel_file():
    p = Path(__file__).parent / "test_data" / "excel_to_xtce_test_file.xlsx"
    return p


def test_generated_xml(excel_file, tmp_path):
    """Make sure we are producing the expected contents within the XML file.

    To produce a new expected output file the following command can be used.
    imap_xtce imap_processing/tests/ccsds/test_data/excel_to_xtce_test_file.xlsx
        --output imap_processing/tests/ccsds/test_data/expected_output.xml
    """
    generator = excel_to_xtce.XTCEGenerator(excel_file)
    output_file = tmp_path / "output.xml"
    generator.to_xml(output_file)

    expected_file = excel_file.parent / "expected_output.xml"
    with open(output_file) as f, open(expected_file) as f_expected:
        assert f.read() == f_expected.read()


# General test
@mock.patch("imap_processing.ccsds.excel_to_xtce.XTCEGenerator")
def test_main_general(mock_input, excel_file):
    """Testing base main function."""
    test_args = [
        "test_script",
        "--output",
        "swe.xml",
        f"{excel_file}",
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

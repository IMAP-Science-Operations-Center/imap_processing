"""Tests for cli module"""
# Installed
import argparse

import pytest

# Local
from imap_processing import cli
from imap_processing.ois import ois_ingest


@pytest.mark.parametrize(
    ("cli_args", "parsed"),
    [
        (
            ["ois-ingest", "--ccsds", "str"],
            argparse.Namespace(func=ois_ingest.ingest, ccsds="str"),
        ),
        (
            ["instr-process", "--instrument", "Ultra", "--level", "L0"],
            argparse.Namespace(instrument="Ultra", level="L0"),
        ),
    ],
)
def test_parse_cli_args(cli_args, parsed):
    """
    Test that cli args are parsed properly
    """
    assert cli._parse_args(cli_args) == parsed

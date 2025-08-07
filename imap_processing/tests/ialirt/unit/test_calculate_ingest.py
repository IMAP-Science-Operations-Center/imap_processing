"""Test processEphemeris functions."""

from datetime import datetime

import numpy as np
import pytest

from imap_processing.ialirt.calculate_ingest import (
    find_tcp_connections,
)
from imap_processing import imap_module_directory

TEST_PATH = imap_module_directory / "tests" / "ialirt" / "data" / "l0"


def test_generate_ingest():
    """
    Test the generate_ingest function.
    """
    filename = "flight_iois_1.log.2025-212T16_55_27.531613"
    # TODO: put this in lambda.
    with open(TEST_PATH / "flight_iois_1.log.2025-212T16_55_27.531613", encoding="utf-8") as f:
        lines = f.readlines()

    test = find_tcp_connections(filename, lines)

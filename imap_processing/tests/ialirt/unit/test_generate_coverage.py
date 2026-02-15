"""Test processEphemeris functions."""

from datetime import datetime, time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from imap_processing.ialirt.generate_coverage import (
    create_schedule_mask,
    format_coverage_summary,
    generate_coverage,
)


@pytest.mark.external_kernel
def test_generate_coverage(furnish_kernels):
    """
    Test the generate_coverage function.
    """
    # Note: tested this code with the Sun and achieved expected
    # results ~12 hours of coverage from horizon to horizon.
    kernels = ["naif0012.tls", "pck00011.tpc", "de440s.bsp", "imap_spk_demo.bsp"]
    with furnish_kernels(kernels):
        coverage_dict, outage_dict = generate_coverage("2026-09-22T00:00:00Z")

    start = datetime.strptime(coverage_dict["Kiel"][0], "%Y-%m-%dT%H:%M:%S.%f")
    end = datetime.strptime(coverage_dict["Kiel"][-1], "%Y-%m-%dT%H:%M:%S.%f")

    duration = end - start
    hours = duration.total_seconds() / 3600

    # Coverage duration should be approximately 9 hours.
    assert hours == pytest.approx(9, abs=1)


@pytest.mark.external_kernel
def test_use_outages(furnish_kernels):
    """
    Test that outages are properly used.
    """
    # Note: tested this code with the Sun and achieved expected
    # results ~12 hours of coverage from horizon to horizon.
    kernels = ["naif0012.tls", "pck00011.tpc", "de440s.bsp", "imap_spk_demo.bsp"]

    outages = {
        "Kiel": [
            ("2026-09-22T11:50:00.00Z", "2026-09-22T12:10:00Z"),
            ("2026-09-22T13:50:00.00Z", "2026-09-22T14:10:00Z"),
            ("2026-09-23T11:50:00.00Z", "2026-09-23T12:10:00Z"),
        ],
    }

    with furnish_kernels(kernels):
        coverage_dict, outage_dict = generate_coverage("2026-09-22T00:00:00Z", outages)

    expected_outages = np.array(
        [
            "2026-09-22T11:50:00.000",
            "2026-09-22T11:55:00.000",
            "2026-09-22T12:00:00.000",
            "2026-09-22T12:05:00.000",
            "2026-09-22T13:50:00.000",
            "2026-09-22T13:55:00.000",
            "2026-09-22T14:00:00.000",
            "2026-09-22T14:05:00.000",
        ]
    )

    assert coverage_dict["Kiel"][0] == "2026-09-22T06:10:00.000"
    assert coverage_dict["Kiel"][-1] == "2026-09-22T16:10:00.000"
    np.testing.assert_array_equal(outage_dict["Kiel"], expected_outages)


@pytest.mark.external_kernel
def test_dsn(furnish_kernels):
    """
    Test that outages are properly used and formatted properly.
    """
    # Note: tested this code with the Sun and achieved expected
    # results ~12 hours of coverage from horizon to horizon.
    kernels = [
        "naif0012.tls",
        "pck00011.tpc",
        "de440s.bsp",
        "imap_spk_demo.bsp",
    ]

    dsn = {
        "DSS-75": [
            ("2026-09-22T11:50:00.00Z", "2026-09-22T14:10:00Z"),
        ]
    }

    outages = {
        "DSS-75": [
            ("2026-09-22T13:50:00.00Z", "2026-09-22T14:10:00Z"),
        ],
    }

    with furnish_kernels(kernels):
        coverage_dict, outage_dict = generate_coverage(
            "2026-09-22T00:00:00Z", outages=outages, dsn=dsn
        )

        assert coverage_dict["DSS-75"][-1] == "2026-09-22T13:45:00.000"

        output = format_coverage_summary(
            coverage_dict, outage_dict, "2026-09-22T00:00:00Z"
        )

        assert "I-ALiRT Coverage Summary" in output["summary"]
        assert 40.6 == output["total_coverage_percent"]


@patch("imap_processing.ialirt.generate_coverage.et_to_utc")
def test_create_schedule_mask(mock_et_to_utc):
    """
    Test create_schedule_mask.
    """

    mock_et_to_utc.return_value = np.array(
        [
            "2026-09-22T11:30:00.000",
            "2026-09-22T11:35:00.000",
            "2026-09-22T11:40:00.000",
            "2026-09-22T11:45:00.000",
            "2026-09-22T11:50:00.000",
            "2026-09-22T11:55:00.000",
            "2026-09-22T12:00:00.000",
            "2026-09-22T12:05:00.000",
            "2026-09-22T12:10:00.000",
            "2026-09-22T12:15:00.000",
            "2026-09-22T12:20:00.000",
            "2026-09-22T12:25:00.000",
            "2026-09-22T12:30:00.000",
        ]
    )

    time_range = np.arange(13)

    station = SimpleNamespace(
        schedule_start=time(12, 0),
        schedule_end=None,
    )

    mask = create_schedule_mask(station, time_range)

    expected = np.array(
        [
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ],
        dtype=bool,
    )

    np.testing.assert_array_equal(mask, expected)

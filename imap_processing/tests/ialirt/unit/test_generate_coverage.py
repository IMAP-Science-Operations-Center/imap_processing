"""Test processEphemeris functions."""

from datetime import datetime

import numpy as np
import pytest

from imap_processing.ialirt.generate_coverage import (
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
    kernels = [
        "pck00011.tpc",
        "de440s.bsp",
    ]
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
    kernels = [
        "pck00011.tpc",
        "de440s.bsp",
    ]

    outages = {
        "Kiel": [
            ("2026-09-22T11:50:00.00Z", "2026-09-22T12:10:00Z"),
            ("2026-09-22T13:50:00.00Z", "2026-09-22T14:10:00Z"),
            ("2026-09-23T11:50:00.00Z", "2026-09-23T12:10:00Z"),
        ],
    }

    with furnish_kernels(kernels):
        coverage_dict, outage_dict = generate_coverage("2026-09-22T00:00:00Z", outages)

    expected = np.array(
        [
            "2026-09-22T07:00:00.000",
            "2026-09-22T08:00:00.000",
            "2026-09-22T09:00:00.000",
            "2026-09-22T10:00:00.000",
            "2026-09-22T11:00:00.000",
            "2026-09-22T13:00:00.000",
            "2026-09-22T15:00:00.000",
            "2026-09-22T16:00:00.000",
        ]
    )
    expected_outages = np.array(["2026-09-22T12:00:00.000", "2026-09-22T14:00:00.000"])

    np.testing.assert_array_equal(coverage_dict["Kiel"], expected)
    np.testing.assert_array_equal(outage_dict["Kiel"], expected_outages)


@pytest.mark.external_kernel
def test_dsn(furnish_kernels):
    """
    Test that outages are properly used and formatted properly.
    """
    # Note: tested this code with the Sun and achieved expected
    # results ~12 hours of coverage from horizon to horizon.
    kernels = [
        "pck00011.tpc",
        "de440s.bsp",
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

    dsn_expected = np.array(["2026-09-22T12:00:00.000", "2026-09-22T13:00:00.000"])
    kiel_expected = np.array(
        [
            "2026-09-22T07:00:00.000",
            "2026-09-22T08:00:00.000",
            "2026-09-22T09:00:00.000",
            "2026-09-22T10:00:00.000",
            "2026-09-22T11:00:00.000",
            "2026-09-22T15:00:00.000",
            "2026-09-22T16:00:00.000",
        ]
    )

    np.testing.assert_array_equal(coverage_dict["Kiel"], kiel_expected)
    np.testing.assert_array_equal(coverage_dict["DSS-75"], dsn_expected)

    output = format_coverage_summary(coverage_dict, outage_dict, "2026-09-22T00:00:00Z")

    assert "I-ALiRT Coverage Summary" in output["summary"]
    assert 37.5 == output["total_coverage_percent"]

"""Test processEphemeris functions."""

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest

from imap_processing.ialirt.constants import STATIONS
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
        assert 42.0 == output["total_coverage_percent"]


@pytest.mark.external_kernel
def test_non_dsn_priority_blocking_with_kernels(furnish_kernels):
    "Test that non-dsn station block other non-dsn stations."
    kernels = ["naif0012.tls", "pck00011.tpc", "de440s.bsp", "imap_spk_demo.bsp"]
    start_time = "2026-09-22T00:00:00Z"

    with furnish_kernels(kernels):
        # Kiel-only coverage
        with patch(
            "imap_processing.ialirt.generate_coverage.NON_DSN_STATIONS",
            new={"Kiel": STATIONS["Kiel"]},
        ):
            coverage_kiel, _ = generate_coverage(start_time)

        kiel_times = coverage_kiel["Kiel"]

        # Manaus-only coverage
        with patch(
            "imap_processing.ialirt.generate_coverage.NON_DSN_STATIONS",
            new={"Manaus": STATIONS["Manaus"]},
        ):
            coverage_manaus_only, _ = generate_coverage(start_time)

        manaus_only_times = coverage_manaus_only["Manaus"]

        overlap = np.intersect1d(kiel_times, manaus_only_times)
        # Assert the times overlap.
        assert overlap.size > 0

        # Kiel first, then Manaus
        with patch(
            "imap_processing.ialirt.generate_coverage.NON_DSN_STATIONS",
            new={
                "Kiel": STATIONS["Kiel"],
                "Manaus": STATIONS["Manaus"],
            },
        ):
            coverage, _ = generate_coverage(start_time)

        manaus_coverage = coverage["Manaus"]

        # Manaus should have no overlap with Kiel.
        blocked_overlap = np.intersect1d(kiel_times, manaus_coverage)
        assert blocked_overlap.size == 0
        assert manaus_coverage[0] > kiel_times[-1]


@pytest.mark.external_kernel
def test_dsn_outage_allows_ground_station_coverage(furnish_kernels):
    """
    DSN contacts block non-DSN stations, but DSN outages remove blocking.
    """
    kernels = ["naif0012.tls", "pck00011.tpc", "de440s.bsp", "imap_spk_demo.bsp"]
    start_time = "2026-09-22T00:00:00Z"

    with furnish_kernels(kernels):
        # Baseline Kiel-only coverage (no DSN)
        with patch(
            "imap_processing.ialirt.generate_coverage.NON_DSN_STATIONS",
            new={"Kiel": STATIONS["Kiel"]},
        ):
            coverage_base, _ = generate_coverage(start_time)

        kiel_times = coverage_base["Kiel"]

        contact_start = kiel_times[10]  # inside Kiel coverage
        contact_end = kiel_times[16]  # 30 min later (inclusive logic in your code)

        # Outage inside the DSN contact
        outage_start = kiel_times[12]
        outage_end = kiel_times[14]

        dsn = {"DSS-75": [(contact_start, contact_end)]}

        # DSN contact, no DSN outage
        # Kiel should be blocked during the full contact window
        with patch(
            "imap_processing.ialirt.generate_coverage.NON_DSN_STATIONS",
            new={"Kiel": STATIONS["Kiel"]},
        ):
            coverage_blocked, _ = generate_coverage(start_time, dsn=dsn, outages=None)

        kiel_blocked = coverage_blocked["Kiel"]

        assert not np.any(np.isin(kiel_times[10:16], kiel_blocked))

        # DSN contact + DSN outage
        # Kiel allowed during outage sub-window
        outages = {"DSS-75": [(outage_start, outage_end)]}

        with patch(
            "imap_processing.ialirt.generate_coverage.NON_DSN_STATIONS",
            new={"Kiel": STATIONS["Kiel"]},
        ):
            coverage_punched, _ = generate_coverage(
                start_time, dsn=dsn, outages=outages
            )

        kiel_punched = coverage_punched["Kiel"]

        assert np.all(np.isin(kiel_times[12:14], kiel_punched))

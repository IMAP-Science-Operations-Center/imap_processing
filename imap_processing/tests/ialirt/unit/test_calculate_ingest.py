"""Test calculate_ingest functions."""

from datetime import datetime, timedelta

from imap_processing import imap_module_directory
from imap_processing.ialirt.calculate_ingest import (
    format_ingest_data,
    packets_created,
)

TEST_PATH = imap_module_directory / "tests" / "ialirt" / "data" / "l0"


def test_packets_created():
    """Test the packets_created function."""
    with open(
        TEST_PATH / "flight_iois_1.log.2026-021T10-58-00.171087", encoding="utf-8"
    ) as f:
        lines = f.readlines()

    actual_output = packets_created(datetime(2026, 7, 31, 16, 33, 39, 0), lines)

    expected = {
        "Kiel": {
            "last_data_received": [
                "2026-01-21T09:57:58Z",
                "2026-01-21T10:27:59Z",
            ],
            "rate_kbps": [2.0, 2.0],
        },
        "UKSA": {
            "last_data_received": [],
            "rate_kbps": [],
        },
        "tlmrelay": {
            "last_data_received": [
                "2026-01-01T00:00:00Z",
            ],
            "rate_kbps": [
                0.0,
            ],
        },
    }

    assert actual_output == expected


def test_format_ingest_data():
    """Test the format_ingest_data function."""
    base_date = datetime.strptime("2025-212", "%Y-%j")
    filenames = []

    for hour in range(24):
        timestamp = (base_date + timedelta(hours=hour)).strftime("%Y-%jT%H_%M_%S.%f")
        filename = f"flight_iois_1.log.{timestamp}"
        filenames.append(filename)

    log_lines = []
    base_date = datetime.strptime("2025-212", "%Y-%j")

    current_time = base_date
    end_time = base_date + timedelta(days=1)

    for _ in range(int((end_time - base_date).total_seconds())):
        time_str = current_time.strftime("%Y/%j-%H:%M:%S.%f")[:-3]

        # Kiel connection window: 08:00 to 16:00
        if current_time == base_date + timedelta(hours=8):
            log_lines.append(f"{time_str} Kiel antenna partner connection is up.\n")
        elif current_time == base_date + timedelta(hours=16):
            log_lines.append(f"{time_str} Kiel antenna partner connection is down!\n")

        # Packet ingest every minute during Kiel connection window: 08:00 to 16:00
        if (
            base_date + timedelta(hours=8)
            <= current_time
            < base_date + timedelta(hours=16)
            and current_time.minute == 0
            and current_time.second == 0
        ):
            pkt_time = current_time.strftime("%Y_%j_%H_%M_%S")
            log_lines.append(
                f"{time_str} Renamed iois_1_packets_{pkt_time}.partial to "
                f"iois_1_packets_{pkt_time}.\n"
            )

        if current_time == base_date + timedelta(hours=8):
            log_lines.append(
                "ID  Description   LastDataRcvd  ConnectionTime  Rate (kbps)\n"
            )
            log_lines.append("10  Kiel          365-08:00:00  365-08:00:00    2.0\n")

        if current_time == base_date + timedelta(hours=15):
            log_lines.append(
                "ID  Description   LastDataRcvd  ConnectionTime  Rate (kbps)\n"
            )
            log_lines.append("10  Kiel          001-15:00:00  001-08:00:00    2.0\n")

        current_time += timedelta(seconds=1)

    filenames = sorted(filenames)

    data = format_ingest_data(filenames[-1], log_lines)

    assert data["Kiel"]["last_data_received"][0] == "2025-12-31T08:00:00Z"
    assert data["Kiel"]["last_data_received"][-1] == "2026-01-01T15:00:00Z"
    assert data["Kiel"]["rate_kbps"][0] == 2.0

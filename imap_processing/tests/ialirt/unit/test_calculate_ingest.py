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

    actual_output = packets_created(datetime(2025, 7, 31, 16, 33, 39, 0), lines)

    # 2025/212-16:33:39.186
    time_0 = datetime(2025, 7, 31, 16, 33, 39, 186000)
    # 2025/212-16:34:40.199
    time_1 = datetime(2025, 7, 31, 16, 34, 40, 199000)

    assert actual_output[0] == time_0
    assert actual_output[1] == time_1


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

        current_time += timedelta(seconds=1)

    filenames = sorted(filenames)

    data = format_ingest_data(filenames[-1], log_lines)

    assert data["packet_ingest"][0] == "2025-07-31T08:00:00"
    assert data["packet_ingest"][-1] == "2025-07-31T15:00:00"
    assert data["connection_times"]["Kiel"][0]["start"] == "2025-07-31T08:00:00"
    assert data["connection_times"]["Kiel"][0]["end"] == "2025-07-31T16:00:00"


def test_format_ingest_data_edge_cases():
    """Test the edge cases of the format_ingest_data function."""

    # File names for a short 3 hour test window
    filenames = [
        "flight_iois_1.log.2025-212T00_00_00.000000",
        "flight_iois_1.log.2025-212T01_00_00.000000",
        "flight_iois_1.log.2025-212T02_00_00.000000",
    ]

    base_date = datetime(2025, 7, 31, 0, 0, 0)
    log_lines = []

    # Simulate case: log starts with a "down!" at 00:15 (no prior "up.")
    timestamp_down = (base_date + timedelta(minutes=15)).strftime("%Y/%j-%H:%M:%S.%f")[
        :-3
    ]
    log_lines.append(f"{timestamp_down} Kiel antenna partner connection is down!\n")

    # Add packet event at 00:00
    timestamp_pkt = base_date.strftime("%Y/%j-%H:%M:%S.%f")[:-3]
    pkt_time = base_date.strftime("%Y_%j_%H_%M_%S")
    log_lines.append(
        f"{timestamp_pkt} Renamed iois_1_packets_{pkt_time}.partial to "
        f"iois_1_packets_{pkt_time}.\n"
    )

    # Simulate case: "up." at 02:00 (no matching "down!" before end of file)
    timestamp_up = (base_date + timedelta(hours=2)).strftime("%Y/%j-%H:%M:%S.%f")[:-3]
    log_lines.append(f"{timestamp_up} Kiel antenna partner connection is up.\n")

    # Add packet event at 02:01
    timestamp_pkt = (base_date + timedelta(hours=2, minutes=1)).strftime(
        "%Y/%j-%H:%M:%S.%f"
    )[:-3]
    pkt_time = (base_date + timedelta(hours=2, minutes=1)).strftime("%Y_%j_%H_%M_%S")
    log_lines.append(
        f"{timestamp_pkt} Renamed iois_1_packets_{pkt_time}.partial to "
        f"iois_1_packets_{pkt_time}.\n"
    )
    filenames = sorted(filenames)

    data = format_ingest_data(filenames[-1], log_lines)

    assert data["connection_times"]["Kiel"][0]["start"] == "2025-07-29T02:00:00"
    assert data["connection_times"]["Kiel"][0]["end"] == "2025-07-31T00:15:00"

    assert data["connection_times"]["Kiel"][1]["start"] == "2025-07-31T02:00:00"
    assert data["connection_times"]["Kiel"][1]["end"] == "2025-07-31T02:00:00"

    assert data["packet_ingest"][0] == "2025-07-31T00:00:00"
    assert data["packet_ingest"][1] == "2025-07-31T02:01:00"

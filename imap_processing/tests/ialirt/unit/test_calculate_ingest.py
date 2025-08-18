"""Test calculate_ingest functions."""

from datetime import datetime, timedelta, timezone
from typing import Any

from imap_processing import imap_module_directory
from imap_processing.ialirt.calculate_ingest import (
    find_tcp_connections,
    format_ingest_data,
    packets_created,
)
from imap_processing.ialirt.constants import STATIONS

TEST_PATH = imap_module_directory / "tests" / "ialirt" / "data" / "l0"


def test_find_tcp_connections():
    """Test the find_tcp_connections function."""
    filename = "flight_iois_1.log.2025-212T16_55_27.531613"
    # File creation time minus 1 hr.
    timestamp_str = filename.split(".")[2]
    timestamp_str = timestamp_str.replace("_", ":")
    start_of_time = datetime.strptime(timestamp_str, "%Y-%jT%H:%M:%S") - timedelta(
        hours=1
    )
    end_of_time = start_of_time + timedelta(hours=48)

    with open(TEST_PATH / filename, encoding="utf-8") as f:
        lines = f.readlines()

    formatted: dict[str, Any] = {
        "summary": "I-ALiRT Real-time Ingest Summary",
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_format": "UTC (ISOC)",
        "stations": list(STATIONS),
        "time_range": [
            start_of_time.isoformat(),
            end_of_time.isoformat(),
        ],  # Overall time range of the data
        "packet_ingest": [],  # Global packet ingest times
        "connection_times": {
            station: [] for station in list(STATIONS)
        },  # Per-station TCP connection windows
    }

    test = find_tcp_connections(start_of_time, end_of_time, lines, formatted)

    # 2025/212-16:33:03.247
    time_0 = datetime(2025, 7, 31, 16, 33, 3, 247000)
    # 2025/212-16:33:40.189
    time_1 = datetime(2025, 7, 31, 16, 33, 40, 189000)

    assert test["connection_times"]["Kiel"][0]["start"] == datetime.isoformat(time_0)
    assert test["connection_times"]["Kiel"][0]["end"] == datetime.isoformat(time_1)


def test_packets_created():
    """Test the packets_created function."""
    with open(
        TEST_PATH / "flight_iois_1.log.2025-212T16_55_27.531613", encoding="utf-8"
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

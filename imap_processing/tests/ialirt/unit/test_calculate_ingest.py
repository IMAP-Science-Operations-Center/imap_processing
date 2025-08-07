"""Test processEphemeris functions."""

from datetime import datetime, timedelta

from imap_processing.ialirt.calculate_ingest import (
    find_tcp_connections,
    format_ingest_data,
    packets_created,
)
from imap_processing import imap_module_directory

TEST_PATH = imap_module_directory / "tests" / "ialirt" / "data" / "l0"


def test_find_tcp_connections():
    """ Test the find_tcp_connections function.
    """
    filename = "flight_iois_1.log.2025-212T16_55_27.531613"
    # TODO: put this in lambda.
    with open(TEST_PATH / filename, encoding="utf-8") as f:
        lines = f.readlines()

    test = find_tcp_connections(filename, lines, "Kiel")

    # 2025/212-16:33:03.247
    time_0 = datetime(2025, 7, 31, 16, 33, 3, 247000)
    # 2025/212-16:33:40.189
    time_1 = datetime(2025, 7, 31, 16, 33, 40, 189000)

    assert test[0][0] == time_0
    assert test[0][1] == time_1


def test_packets_created():
    """ Test the packets_created function.
    """
    # TODO: put this in lambda.
    with open(TEST_PATH / "flight_iois_1.log.2025-212T16_55_27.531613", encoding="utf-8") as f:
        lines = f.readlines()

    test = packets_created(lines)

    # 2025/212-16:33:39.186
    time_0 = datetime(2025, 7, 31, 16, 33, 39, 186000)
    # 2025/212-16:34:40.199
    time_1 = datetime(2025, 7, 31, 16, 34, 40, 199000)

    assert test[0] == time_0
    assert test[1] == time_1


def test_format_ingest_data():
    """ Test the format_ingest_data function.
    """
    base_date = datetime.strptime("2025-212", "%Y-%j")  # 2025-07-31
    filenames = []

    for hour in range(24):
        timestamp = (base_date + timedelta(hours=hour)).strftime("%Y-%jT%H_%M_%S.%f")
        filename = f"flight_iois_1.log.{timestamp}"
        filenames.append(filename)

    log_lines = []
    base_date = datetime.strptime("2025-212", "%Y-%j")

    # Simulate every second of the day (optional: make sparser)
    current_time = base_date
    end_time = base_date + timedelta(days=1)

    for i in range(int((end_time - base_date).total_seconds())):
        time_str = current_time.strftime("%Y/%j-%H:%M:%S.%f")[:-3]

        # Kiel connection window: 08:00 – 16:00
        if current_time == base_date + timedelta(hours=8):
            log_lines.append(f"{time_str} Kiel antenna partner connection is up.\n")
        elif current_time == base_date + timedelta(hours=16):
            log_lines.append(f"{time_str} Kiel antenna partner connection is down!\n")

        # Packet ingest every minute during Kiel connection window: 08:00–16:00
        if (
                base_date + timedelta(hours=8) <= current_time < base_date + timedelta(hours=16)
                and current_time.minute == 0
                and current_time.second == 0
        ):
            pkt_time = current_time.strftime("%Y_%j_%H_%M_%S")
            log_lines.append(
                f"{time_str} Renamed iois_1_packets_{pkt_time}.partial to iois_1_packets_{pkt_time}.\n"
            )

        current_time += timedelta(seconds=1)

    data = format_ingest_data(filenames, log_lines)

    assert data["packet_ingest"][0] == "2025-07-31T08:00:00"
    assert data["packet_ingest"][-1] == "2025-07-31T15:00:00"
    assert data["tcp"]["Kiel"][0]["start"] == "2025-07-31T08:00:00"
    assert data["tcp"]["Kiel"][0]["end"] == "2025-07-31T16:00:00"


def test_format_ingest_data_edge_cases():
    """ Test the format_ingest_data_edge_cases function.
    """

    # File names for a short 3-hour test window
    filenames = [
        "flight_iois_1.log.2025-212T00_00_00.000000",
        "flight_iois_1.log.2025-212T01_00_00.000000",
        "flight_iois_1.log.2025-212T02_00_00.000000",
    ]

    base_date = datetime(2025, 7, 31, 0, 0, 0)
    log_lines = []

    # Simulate case: log starts with a "down!" at 00:15 (no prior "up.")
    ts_down = (base_date + timedelta(minutes=15)).strftime("%Y/%j-%H:%M:%S.%f")[:-3]
    log_lines.append(f"{ts_down} Kiel antenna partner connection is down!\n")

    # Add packet event between 00:00 and 00:15 (while we assume connection was active)
    ts_pkt = base_date.strftime("%Y/%j-%H:%M:%S.%f")[:-3]
    pkt_time = base_date.strftime("%Y_%j_%H_%M_%S")
    log_lines.append(
        f"{ts_pkt} Renamed iois_1_packets_{pkt_time}.partial to iois_1_packets_{pkt_time}.\n"
    )

    # Simulate case: "up." at 02:00 (no matching "down!" before end of file)
    ts_up = (base_date + timedelta(hours=2)).strftime("%Y/%j-%H:%M:%S.%f")[:-3]
    log_lines.append(f"{ts_up} Kiel antenna partner connection is up.\n")

    # Add packet event at 02:01 (valid, during assumed open connection)
    ts_pkt_2 = (base_date + timedelta(hours=2, minutes=1)).strftime("%Y/%j-%H:%M:%S.%f")[:-3]
    pkt_time_2 = (base_date + timedelta(hours=2, minutes=1)).strftime("%Y_%j_%H_%M_%S")
    log_lines.append(
        f"{ts_pkt_2} Renamed iois_1_packets_{pkt_time_2}.partial to iois_1_packets_{pkt_time_2}.\n"
    )
    filenames = sorted(filenames)

    # Run formatting function
    data = format_ingest_data(filenames[0], filenames[-1], log_lines)

    assert data["tcp"]["Kiel"][0]["start"] == "2025-07-30T23:00:00"
    assert data["tcp"]["Kiel"][0]["end"] == "2025-07-31T00:15:00"

    assert data["tcp"]["Kiel"][1]["start"] == "2025-07-31T02:00:00"
    assert data["tcp"]["Kiel"][1]["end"] == "2025-07-31T02:00:00"

    assert data["packet_ingest"][0] == "2025-07-31T00:00:00"
    assert data["packet_ingest"][1] == "2025-07-31T02:01:00"

    # View output if needed
    import json
    print(json.dumps(data, indent=2))
    print('hi')

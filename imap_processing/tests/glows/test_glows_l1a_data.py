import ast
import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from imap_processing.glows import __version__
from imap_processing.glows.l0 import decom_glows
from imap_processing.glows.l1.glows_l1a import process_de_l0
from imap_processing.glows.l1.glows_l1a_data import (
    DirectEventL1A,
    HistogramL1A,
    StatusData,
)
from imap_processing.glows.utils.constants import DirectEvent, TimeTuple


@pytest.fixture(scope="module")
def decom_test_data():
    """Read test data from file"""
    current_directory = Path(__file__).parent
    packet_path = current_directory / "glows_test_packet_20110921_v01.pkts"
    data_packet_list = decom_glows.decom_packets(packet_path)
    return data_packet_list


@pytest.fixture()
def l1a_test_data(decom_test_data):
    hist_l1a = []

    for hist in decom_test_data[0]:
        hist_l1a.append(HistogramL1A(hist))

    de_l1a_dict = process_de_l0(decom_test_data[1])

    de_l1a = []
    for _, value in de_l1a_dict.items():
        de_l1a += value

    return hist_l1a, de_l1a


@pytest.fixture()
def histogram_test_data(decom_test_data):
    histl0 = decom_test_data[0][0]
    hist = HistogramL1A(histl0)
    return hist


@pytest.fixture()
def de_test_data(decom_test_data):
    del0 = decom_test_data[1][0]
    de = DirectEventL1A(del0)
    return de


def test_histogram_list(histogram_test_data, decom_test_data):
    """Test size of histogram data"""
    histl0 = decom_test_data[0][0]

    assert len(histogram_test_data.histograms) == 3600
    assert sum(histogram_test_data.histograms) == histl0.EVENTS


def test_histogram_attributes(histogram_test_data):
    """Test other data in histogram packet"""

    expected_block_header = {
        "flight_software_version": 131329,
        "ground_software_version": __version__,
        "pkts_file_name": "glows_test_packet_20110921_v01.pkts",
        "seq_count_in_pkts_file": 0,
    }

    assert histogram_test_data.flight_software_version == expected_block_header["flight_software_version"]
    assert histogram_test_data.ground_software_version == expected_block_header["ground_software_version"]
    assert histogram_test_data.pkts_file_name == expected_block_header["pkts_file_name"]
    assert histogram_test_data.seq_count_in_pkts_file == expected_block_header["seq_count_in_pkts_file"]
    assert histogram_test_data.last_spin_id == 0

    assert histogram_test_data.imap_start_time == TimeTuple(54232215, 0)
    assert histogram_test_data.imap_time_offset == TimeTuple(120, 0)
    assert histogram_test_data.glows_start_time == TimeTuple(54232214, 1997263)
    assert histogram_test_data.glows_time_offset == TimeTuple(119, 1998758)
    assert histogram_test_data.flags == {
        "flags_set_onboard": 64,
        "is_generated_on_ground": False,
    }


def test_direct_event_every_second(de_test_data):
    """Test that Level 1A direct events generate correct status data table"""

    # Manually created dictionary from first test packet
    expected_data_every_second = {
        "imap_sclk_last_pps": 54232338,
        "glows_sclk_last_pps": 54232337,
        "glows_ssclk_last_pps": 1995990,
        "imap_sclk_next_pps": 54232339,
        "catbed_heater_active": 0,
        "spin_period_valid": 1,
        "spin_phase_at_next_pps_valid": 1,
        "spin_period_source": 0,
        "spin_period": 46875,
        "spin_phase_at_next_pps": 17476,
        "number_of_completed_spins": 3615489,
        "filter_temperature": 2507,
        "hv_voltage": 2007,
        "glows_time_on_pps_valid": 1,
        "time_status_valid": 1,
        "housekeeping_valid": 1,
        "is_pps_autogenerated": 0,
        "hv_test_in_progress": 0,
        "pulse_test_in_progress": 0,
        "memory_error_detected": 0,
    }

    assert dataclasses.asdict(de_test_data.status_data) == expected_data_every_second


def test_direct_events_uncompressed(de_test_data):
    """test building of uncompressed events"""

    first_uncompressed_event = bytearray.fromhex("033b8511061e7bf0")
    processed_uncompressed = de_test_data._build_uncompressed_event(
        first_uncompressed_event
    )
    expected = DirectEvent(
        TimeTuple(54232337, 1997808), impulse_length=6, multi_event=False
    )
    assert processed_uncompressed == expected


def test_direct_events_two_bytes_compressed(de_test_data):
    """Test two byte compression for direct events"""

    first_byte = int("0x87", 0)
    two_bytes = bytearray.fromhex("6106")

    expected = DirectEvent(
        TimeTuple(seconds=54232337, subseconds=1999697),
        impulse_length=6,
        multi_event=False,
    )
    # Required for processing
    previous_time = TimeTuple(54232337, 1997808)
    oldest_diff = first_byte & 0x3F

    output = de_test_data._build_compressed_event(two_bytes, oldest_diff, previous_time)
    second = de_test_data._build_compressed_event(
        bytearray.fromhex("d806"), 7, output.timestamp
    )

    second_expected = DirectEvent(
        TimeTuple(seconds=54232338, subseconds=1705),
        impulse_length=6,
        multi_event=False,
    )

    assert expected == output
    assert expected == de_test_data.direct_events[1]
    assert second_expected == second


def test_direct_events_three_bytes_compressed(de_test_data):
    """test 3 byte compression for direct events"""
    first_byte = int("0xE7", 0)
    three_bytes = bytearray.fromhex("620687")

    expected = DirectEvent(
        TimeTuple(54232339, 578806),
        impulse_length=135,
        multi_event=False,
    )

    # Required for processing
    previous_time = TimeTuple(54232337, 1997808)

    oldest_diff = first_byte & 0x3F

    output = de_test_data._build_compressed_event(
        three_bytes, oldest_diff, previous_time
    )

    assert expected == output

    bad_input = bytearray.fromhex("11111111")
    with pytest.raises(ValueError, match="Incorrect length"):
        de_test_data._build_compressed_event(bad_input, oldest_diff, previous_time)


def test_sequential_direct_events(decom_test_data):
    """test output of the first level 0 direct event packet"""

    # First, an uncompressed event, then 3 2-bit compressed events (876106), then 3
    # 3-bit compressed events (E7620687)
    test_list = bytearray.fromhex(
        "033b8511061e7bf0"  # uncompressed
        "876106876106876106"  # 2 bit compressed
        "E7620687E7620687E7620687"
    )  # 3 bit compressed

    de_second_val = DirectEventL1A(decom_test_data[1][1])

    list_events = de_second_val._generate_direct_events(test_list)

    assert len(list_events) == 7

    first_expected = DirectEvent(
        TimeTuple(seconds=54232338, subseconds=1996127),
        impulse_length=6,
        multi_event=False,
    )

    final_expected = DirectEvent(
        TimeTuple(seconds=54232339, subseconds=1993947),
        impulse_length=6,
        multi_event=False,
    )
    second_to_last = DirectEvent(
        TimeTuple(seconds=54232339, subseconds=1990921),
        impulse_length=6,
        multi_event=False,
    )
    assert de_second_val.direct_events[0] == first_expected
    assert len(de_second_val.direct_events) == 721
    assert de_second_val.direct_events[-1] == final_expected
    assert de_second_val.direct_events[-2] == second_to_last


def test_generate_direct_events(de_test_data):
    """test multiple types of compression

    first byte uncompressed, then 2 byte compressed, then 3 byte compressed"""
    first_uncompressed_event = bytearray.fromhex("033b8511061e7bf0")
    two_bytes = bytearray.fromhex("876106")
    three_bytes = bytearray.fromhex("E7620687")

    test_data = first_uncompressed_event + two_bytes + three_bytes
    output = de_test_data._generate_direct_events(test_data)

    expected = [
        DirectEvent(TimeTuple(54232337, 1997808), impulse_length=6, multi_event=False),
        DirectEvent(
            TimeTuple(54232337, 1999697),
            impulse_length=6,
            multi_event=False,
        ),
        DirectEvent(
            TimeTuple(54232339, 580695),
            impulse_length=135,
            multi_event=False,
        ),
    ]

    assert output == expected


def test_combine_direct_events(decom_test_data):
    """Test having multiple direct events across different level 0 instances"""
    de0 = decom_test_data[1][0]
    de0_first = dataclasses.replace(de0, LEN=2, SEQ=0, DE_DATA=de0.DE_DATA[:100])
    de0_second = dataclasses.replace(de0, LEN=2, SEQ=1, DE_DATA=de0.DE_DATA[100:])

    expected = DirectEventL1A(de0)

    de1 = DirectEventL1A(de0_first)
    assert not de1.direct_events
    if not de1.direct_events:
        de1.append(de0_second)

    assert de1.direct_events

    # l0 attribute should not be equal, but everything else should be
    assert de1.de_data == expected.de_data
    assert de1.most_recent_seq == 1
    assert de1.missing_seq == []
    assert de1.status_data == expected.status_data
    assert de1.direct_events == expected.direct_events


def test_combine_direct_events_with_missing(decom_test_data):
    """test for missing direct events in the sequence"""
    de0 = decom_test_data[1][0]
    de0_first = dataclasses.replace(de0, LEN=3, SEQ=0, DE_DATA=de0.DE_DATA[:100])
    de0_second = dataclasses.replace(de0, LEN=3, SEQ=2, DE_DATA=de0.DE_DATA[100:])

    expected = DirectEventL1A(de0)

    de1 = DirectEventL1A(de0_first)
    assert not de1.direct_events
    if not de1.direct_events:
        de1.append(de0_second)

    assert de1.direct_events
    # Missing one value
    assert de1.missing_seq == [1]
    assert de1.direct_events == expected.direct_events


def test_generate_status_data():
    """Test if status data is correctly generated from 40 byte input"""
    input_bytes = bytearray.fromhex(
        "00000010"
        "0007F737"
        "0007FAF9"
        "00004330"
        "01"
        "01"
        "00"
        "01"
        "03DE"
        "08F3"
        "0022F917"
        "1DE6"
        "0076"
        "0100"
        "01"
        "01"
        "00"
        "00"
        "01"
        "00"
    )

    # completely fabricated
    test_data = {
        "imap_sclk_last_pps": 16,
        "glows_sclk_last_pps": 522039,
        "glows_ssclk_last_pps": 523001,
        "imap_sclk_next_pps": 17200,
        "catbed_heater_active": 1,
        "spin_period_valid": 1,
        "spin_phase_at_next_pps_valid": 0,
        "spin_period_source": 1,
        "spin_period": 990,
        "spin_phase_at_next_pps": 2291,
        "number_of_completed_spins": 2291991,
        "filter_temperature": 7654,
        "hv_voltage": 118,
        "glows_time_on_pps_valid": 1,
        "time_status_valid": 0,
        "housekeeping_valid": 1,
        "is_pps_autogenerated": 1,
        "hv_test_in_progress": 0,
        "pulse_test_in_progress": 0,
        "memory_error_detected": 1,
    }

    output = StatusData(input_bytes)
    assert dataclasses.asdict(output) == test_data

    # real data
    # From first test packet, first 40 bytes
    test2 = bytearray.fromhex(
        "033b8512033b8511001e74d6033b851300010100b71b444400372b0109cb07d70101010000000000"
    )

    expected = {
        "imap_sclk_last_pps": 54232338,
        "glows_sclk_last_pps": 54232337,
        "glows_ssclk_last_pps": 1995990,
        "imap_sclk_next_pps": 54232339,
        "catbed_heater_active": 0,
        "spin_period_valid": 1,
        "spin_phase_at_next_pps_valid": 1,
        "spin_period_source": 0,
        "spin_period": 46875,
        "spin_phase_at_next_pps": 17476,
        "number_of_completed_spins": 3615489,
        "filter_temperature": 2507,
        "hv_voltage": 2007,
        "glows_time_on_pps_valid": 1,
        "time_status_valid": 1,
        "housekeeping_valid": 1,
        "is_pps_autogenerated": 0,
        "hv_test_in_progress": 0,
        "pulse_test_in_progress": 0,
        "memory_error_detected": 0,
    }

    output = StatusData(test2)
    assert dataclasses.asdict(output) == expected


def test_expected_results(l1a_test_data):
    _, de_data = l1a_test_data

    # Validation data is generated from the code sent over by GLOWS team. Contains the
    # first 20 packets
    validation_data = pd.read_csv(
        Path(__file__).parent / "direct_events_validation_data_l1a.csv"
    )

    for index in validation_data.index:
        de = de_data[validation_data["packet_counter"][index]]

        assert (
            de.l0.ccsds_header.SRC_SEQ_CTR
            == validation_data["seq_count_in_pkts_file"][index]
        )
        assert (
            de.status_data.imap_sclk_last_pps
            == validation_data["imap_sclk_last_pps"][index]
        )
        assert (
            de.status_data.glows_sclk_last_pps
            == validation_data["glows_sclk_last_pps"][index]
        )
        assert (
            de.status_data.glows_ssclk_last_pps
            == validation_data["glows_ssclk_last_pps"][index]
        )
        assert (
            de.status_data.imap_sclk_next_pps
            == validation_data["imap_sclk_next_pps"][index]
        )
        assert (
            de.status_data.catbed_heater_active
            == validation_data["catbed_heater_active"][index]
        )
        assert (
            de.status_data.spin_period_valid
            == validation_data["spin_period_valid"][index]
        )
        assert (
            de.status_data.spin_phase_at_next_pps_valid
            == validation_data["spin_phase_at_next_pps_valid"][index]
        )
        assert (
            de.status_data.spin_period_source
            == validation_data["spin_period_source"][index]
        )
        assert de.status_data.spin_period == validation_data["spin_period"][index]
        assert (
            de.status_data.spin_phase_at_next_pps
            == validation_data["spin_phase_at_next_pps"][index]
        )
        assert (
            de.status_data.number_of_completed_spins
            == validation_data["number_of_completed_spins"][index]
        )
        assert (
            de.status_data.filter_temperature
            == validation_data["filter_temperature"][index]
        )
        assert de.status_data.hv_voltage == validation_data["hv_voltage"][index]
        assert (
            de.status_data.glows_time_on_pps_valid
            == validation_data["glows_time_on_pps_valid"][index]
        )
        assert (
            de.status_data.time_status_valid
            == validation_data["time_status_valid"][index]
        )
        assert (
            de.status_data.housekeeping_valid
            == validation_data["housekeeping_valid"][index]
        )
        assert (
            de.status_data.is_pps_autogenerated
            == validation_data["is_pps_autogenerated"][index]
        )
        assert (
            de.status_data.hv_test_in_progress
            == validation_data["hv_test_in_progress"][index]
        )
        assert (
            de.status_data.pulse_test_in_progress
            == validation_data["pulse_test_in_progress"][index]
        )
        assert (
            de.status_data.memory_error_detected
            == validation_data["memory_error_detected"][index]
        )

        assert de.l0.LEN == validation_data["number_of_de_packets"][index]

        validation_de = ast.literal_eval(validation_data["de_data"][index])

        assert (
            de.direct_events[
                validation_data["de_data_counter"][index]
            ].timestamp.seconds
            == validation_de[0]
        )
        assert (
            de.direct_events[
                validation_data["de_data_counter"][index]
            ].timestamp.subseconds
            == validation_de[1]
        )
        assert (
            de.direct_events[validation_data["de_data_counter"][index]].impulse_length
            == validation_de[2]
        )
        assert (
            de.direct_events[validation_data["de_data_counter"][index]].multi_event
            == validation_de[3]
        )

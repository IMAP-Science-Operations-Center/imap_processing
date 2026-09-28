"""Reproduce the manual verification of raw ``iois_1_packets_*`` capture files.

These are ad hoc telemetry captures that live locally in ``~/Downloads`` and
are not part of the repo's committed/external test data set, so the
``xarray_data`` fixture skips the whole module if a file isn't present on
disk.

Two things are checked, matching the manual verification:

* Header consistency: every packet shares the same APID and CCSDS header
  constants ("expected sync pattern"), and the same packet-length field.
* Source sequence counter: any gap in ``src_seq_ctr`` is fully explained by
  elapsed seconds on the onboard clock (``sc_sclk_sec``), accounting for
  14-bit rollover -- i.e. gaps reflect real missing/uncaptured packets, not
  counter corruption.
"""

from pathlib import Path

import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.spice.time import met_to_utc
from imap_processing.utils import packet_file_to_datasets

IALIRT_XTCE_PATH = (
    imap_module_directory / "ialirt" / "packet_definitions" / "ialirt.xml"
)
DOWNLOADS_DIR = Path.home() / "Downloads"

SC_APID = 478
EXPECTED_PACKET_DATA_LENGTH = 176  # PKT_LEN field value for the SC (apid 478) packet
SEQ_CTR_MAX = 16384  # 14-bit rollover
SUB_SEC_CONVERSION = 256  # fine-time units per second for SC_SCLK_SUB_SEC

PACKET_FILES = [
    "iois_1_packets_2026_268_21_14_27",
    "iois_1_packets_2026_268_21_15_28",
    "iois_1_packets_2026_268_21_16_29",
    "iois_1_packets_2026_268_21_17_30",
    "iois_1_packets_2026_268_21_18_31",
    "iois_1_packets_2026_268_21_19_32",
    "iois_1_packets_2026_268_21_20_33",
    "iois_1_packets_2026_268_21_21_34",
    "iois_1_packets_2026_268_21_22_35",
    "iois_1_packets_2026_268_21_23_36",
    "iois_1_packets_2026_268_21_24_37",
    "iois_1_packets_2026_268_21_25_38",
    "iois_1_packets_2026_268_21_26_39",
    "iois_1_packets_2026_268_21_28_41",
    "iois_1_packets_2026_268_21_29_42",
    "iois_1_packets_2026_268_21_30_43",
    "iois_1_packets_2026_268_21_31_44",
    "iois_1_packets_2026_268_21_32_45",
    "iois_1_packets_2026_268_21_33_46",
    "iois_1_packets_2026_268_21_34_47",
    "iois_1_packets_2026_268_21_35_48",
    "iois_1_packets_2026_268_21_36_49",
    "iois_1_packets_2026_268_21_37_50",
    "iois_1_packets_2026_268_21_38_51",
    "iois_1_packets_2026_268_21_39_52",
    "iois_1_packets_2026_268_21_40_54",
]


def _packet_path(filename: str) -> Path:
    path = DOWNLOADS_DIR / filename
    if not path.exists():
        pytest.skip(f"{path} not found locally")
    return path


@pytest.fixture
def xarray_data():
    """Create merged xarray data for all captured SC (apid 478) packets."""
    packet_paths = tuple(_packet_path(filename) for filename in PACKET_FILES)

    xarray_data = tuple(
        packet_file_to_datasets(packet, IALIRT_XTCE_PATH, use_derived_value=False)[
            SC_APID
        ]
        for packet in packet_paths
    )

    merged_xarray_data = xr.concat(xarray_data, dim="epoch").sortby("epoch")
    return merged_xarray_data


def test_header_fields_are_consistent(xarray_data):
    """Check APID, CCSDS header constants, and packet length."""
    assert len(xarray_data["epoch"]) > 0

    assert set(xarray_data["pkt_apid"].values.tolist()) == {SC_APID}
    assert set(xarray_data["version"].values.tolist()) == {0}
    assert set(xarray_data["type"].values.tolist()) == {0}
    assert set(xarray_data["sec_hdr_flg"].values.tolist()) == {1}
    assert set(xarray_data["pkt_len"].values.tolist()) == {EXPECTED_PACKET_DATA_LENGTH}


def _find_sequence_counter_gaps(seq_counters) -> list[tuple[int, int]]:
    """Find every place src_seq_ctr does not increment by exactly 1.

    This is the naive check (same logic as
    ``imap_processing.utils._check_source_sequence_counter``) that first
    surfaced the gaps -- it does not know about elapsed clock time, so a gap
    here just means "not (previous + 1) % SEQ_CTR_MAX", real dropped packet
    or not.
    """
    gaps = []
    for i in range(1, len(seq_counters)):
        expected_next = (int(seq_counters[i - 1]) + 1) % SEQ_CTR_MAX
        if seq_counters[i] != expected_next:
            gaps.append((int(seq_counters[i - 1]), int(seq_counters[i])))
    return gaps


def test_sequence_counter_gaps_match_elapsed_clock_time(xarray_data):
    """Check that src_seq_ctr jumps are explained by elapsed sc_sclk_sec."""
    seq_counters = xarray_data["src_seq_ctr"].values
    sclk_seconds = xarray_data["sc_sclk_sec"].values

    # sc_sclk_sec alone is only the coarse (integer-second) spacecraft clock
    # count -- valid for elapsed-time math on its own, but not a real
    # calendar time. Combine it with the fine time (sc_sclk_sub_sec) and run
    # it through the SCLK kernel (via met_to_utc) to get an actual UTC
    # timestamp for each packet.
    met = calculate_time(
        xarray_data["sc_sclk_sec"], xarray_data["sc_sclk_sub_sec"], SUB_SEC_CONVERSION
    )
    utc_times = met_to_utc(met.values)

    gaps = _find_sequence_counter_gaps(seq_counters)
    print(f"gaps = {gaps}")

    # Every gap found above should be fully explained by elapsed seconds on
    # the onboard clock (accounting for 14-bit rollover) -- i.e. it reflects
    # real missing/uncaptured packets, not counter corruption.
    for i in range(1, len(seq_counters)):
        elapsed_seconds = int(sclk_seconds[i]) - int(sclk_seconds[i - 1])
        expected_seq_ctr = (int(seq_counters[i - 1]) + elapsed_seconds) % SEQ_CTR_MAX
        assert seq_counters[i] == expected_seq_ctr, (
            f"Packet {i}: src_seq_ctr {seq_counters[i]} does not match "
            f"expected {expected_seq_ctr} given {elapsed_seconds}s elapsed "
            f"on sc_sclk_sec"
        )
        if seq_counters[i] != (int(seq_counters[i - 1]) + 1) % SEQ_CTR_MAX:
            print(
                f"  gap: seq {seq_counters[i - 1]} ({utc_times[i - 1]} UTC) -> "
                f"{seq_counters[i]} ({utc_times[i]} UTC)"
            )

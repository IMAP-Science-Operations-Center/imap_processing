from collections import namedtuple

import numpy as np
import pytest

from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.lo.l0.data_classes.science_counts import ScienceCounts
from imap_processing.lo.l0.data_classes.science_direct_events import ScienceDirectEvents
from imap_processing.lo.l1a.lo_data_container import LoContainer
from imap_processing.lo.l1a.lo_l1a_write_cdfs import (
    create_lo_scicnt_dataset,
    create_lo_scide_dataset,
    write_lo_l1a_cdfs,
)


@pytest.fixture()
def direct_events():
    fake_data_field = namedtuple("fake_packet", ["raw_value", "derived_value"])
    scide1 = ScienceDirectEvents.__new__(ScienceDirectEvents)
    scide2 = ScienceDirectEvents.__new__(ScienceDirectEvents)
    scide1.ccsds_header = CcsdsData(
        {
            "VERSION": fake_data_field(0, 0),
            "TYPE": fake_data_field(0, 0),
            "SEC_HDR_FLG": fake_data_field(0, 0),
            "PKT_APID": fake_data_field(706, 706),
            "SEQ_FLGS": fake_data_field(0, 0),
            "SRC_SEQ_CTR": fake_data_field(0, 0),
            "PKT_LEN": fake_data_field(0, 0),
        }
    )

    scide2.ccsds_header = CcsdsData(
        {
            "VERSION": fake_data_field(0, 0),
            "TYPE": fake_data_field(0, 0),
            "SEC_HDR_FLG": fake_data_field(0, 0),
            "PKT_APID": fake_data_field(706, 706),
            "SEQ_FLGS": fake_data_field(0, 0),
            "SRC_SEQ_CTR": fake_data_field(0, 0),
            "PKT_LEN": fake_data_field(0, 0),
        }
    )
    scide1.TIME = np.array([1712787078, 1712787090, 1712787100])
    scide1.ENERGY = np.array([0, 0, 0])
    scide1.MODE = np.array([0, 0, 0])
    scide1.TOF0 = np.array([0, 0, 0])
    scide1.TOF1 = np.array([0, 0, 0])
    scide1.TOF2 = np.array([0, 0, 0])
    scide1.TOF3 = np.array([0, 0, 0])
    scide1.CKSM = np.array([0, 0, 0])
    scide1.POS = np.array([0, 0, 0])

    scide2.TIME = np.array([1712787150, 1712787170, 1712787200])
    scide2.ENERGY = np.array([0, 0, 0])
    scide2.MODE = np.array([0, 0, 0])
    scide2.TOF0 = np.array([0, 0, 0])
    scide2.TOF1 = np.array([0, 0, 0])
    scide2.TOF2 = np.array([0, 0, 0])
    scide2.TOF3 = np.array([0, 0, 0])
    scide2.CKSM = np.array([0, 0, 0])
    scide2.POS = np.array([0, 0, 0])

    return [scide1, scide2]


@pytest.fixture()
def science_counts():
    fake_data_field = namedtuple("fake_packet", ["raw_value", "derived_value"])
    scicnt1 = ScienceCounts.__new__(ScienceCounts)
    scicnt2 = ScienceCounts.__new__(ScienceCounts)
    scicnt1.ccsds_header = CcsdsData(
        {
            "VERSION": fake_data_field(0, 0),
            "TYPE": fake_data_field(0, 0),
            "SEC_HDR_FLG": fake_data_field(0, 0),
            "PKT_APID": fake_data_field(706, 706),
            "SEQ_FLGS": fake_data_field(0, 0),
            "SRC_SEQ_CTR": fake_data_field(0, 0),
            "PKT_LEN": fake_data_field(0, 0),
        }
    )

    scicnt2.ccsds_header = CcsdsData(
        {
            "VERSION": fake_data_field(0, 0),
            "TYPE": fake_data_field(0, 0),
            "SEC_HDR_FLG": fake_data_field(0, 0),
            "PKT_APID": fake_data_field(706, 706),
            "SEQ_FLGS": fake_data_field(0, 0),
            "SRC_SEQ_CTR": fake_data_field(0, 0),
            "PKT_LEN": fake_data_field(0, 0),
        }
    )
    fields = [
        "START_A",
        "START_C",
        "STOP_B0",
        "STOP_B3",
        "TOF0",
        "TOF1",
        "TOF2",
        "TOF3",
        "TOF0_TOF1",
        "TOF0_TOF2",
        "TOF1_TOF2",
        "SILVER",
        "DISC_TOF0",
        "DISC_TOF1",
        "DISC_TOF2",
        "DISC_TOF3",
        "POS0",
        "POS1",
        "POS2",
        "POS3",
        "HYDROGEN",
        "OXYGEN",
    ]

    scicnt1.SHCOARSE = np.array([1712787078, 1712787090, 1712787100])
    scicnt2.SHCOARSE = np.array([1712787150, 1712787170, 1712787200])
    for field in fields:
        setattr(scicnt1, field, np.array([0, 0, 0]))
        setattr(scicnt2, field, np.array([0, 0, 0]))

    return [scicnt1, scicnt2]


def test_create_lo_scide_dataset(direct_events):
    dataset = create_lo_scide_dataset(direct_events)

    assert (
        dataset["de_time"].data
        == [1712787078, 1712787090, 1712787100, 1712787150, 1712787170, 1712787200]
    ).all()
    assert (dataset["energy"].data == [0, 0, 0, 0, 0, 0]).all()
    assert (dataset["mode"].data == [0, 0, 0, 0, 0, 0]).all()
    assert (dataset["tof0"].data == [0, 0, 0, 0, 0, 0]).all()
    assert (dataset["tof1"].data == [0, 0, 0, 0, 0, 0]).all()
    assert (dataset["tof2"].data == [0, 0, 0, 0, 0, 0]).all()
    assert (dataset["tof3"].data == [0, 0, 0, 0, 0, 0]).all()
    assert (dataset["checksum"].data == [0, 0, 0, 0, 0, 0]).all()
    assert (dataset["pos"].data == [0, 0, 0, 0, 0, 0]).all()


def test_create_lo_scicnt_dataset(science_counts):
    dataset = create_lo_scicnt_dataset(science_counts)

    fields = [
        "start_a",
        "stop_b0",
        "stop_b3",
        "tof0",
        "tof1",
        "tof2",
        "tof3",
        "tof0_tof1",
        "tof0_tof2",
        "tof1_tof2",
        "silver",
        "disc_tof0",
        "disc_tof1",
        "disc_tof2",
        "disc_tof3",
        "pos0",
        "pos1",
        "pos2",
        "pos3",
        "hydrogen",
        "oxygen",
    ]

    expected_epoch = np.array(
        np.array(
            [1712787078, 1712787090, 1712787100, 1712787150, 1712787170, 1712787200]
        )
        * 1e9,
        dtype="datetime64[ns]",
    )

    assert (dataset["epoch"].data == [time for time in expected_epoch]).all()

    for field in fields:
        assert (dataset[field].data == [0, 0, 0, 0, 0, 0]).all()


def test_write_lo_l1a_cdfs(direct_events):
    lo_data = LoContainer()
    lo_data.add(direct_events[0])
    lo_data.add(direct_events[1])

    created_file_paths = write_lo_l1a_cdfs(lo_data)

    assert created_file_paths[0].name == "imap_lo_l1a_scide_20240410_v001.cdf"

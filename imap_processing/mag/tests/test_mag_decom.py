import dataclasses
from pathlib import Path

from imap_processing.mag.l0.decom_mag import decom_packets


def test_burst_decom():
    current_directory = Path(__file__).parent
    burst_test_file = current_directory / "mag_single_burst.pkts"
    l0 = decom_packets(str(burst_test_file))[0]

    assert l0.ccsds_header.PKT_APID == 1068
    assert l0.ccsds_header.SRC_SEQ_CTR == 1
    assert l0.ccsds_header.PKT_LEN == 1821

    expected = dataclasses.replace(
        l0,
        SHCOARSE=435954632,
        PUS_STYPE=21,
        PUS_SSUBTYPE=3,
        COMPRESSION=0,
        MAGO_ACT=1,
        MAGI_ACT=1,
        PRI_SENS=0,
        PRI_VECSEC=6,
        SEC_VECSEC=3,
        PRI_COARSETM=435954625,
        PRI_FNTM=64610,
        SEC_COARSETM=435954625,
        SEC_FNTM=64607,
    )
    assert l0 == expected


def test_norm_decom():
    current_directory = Path(__file__).parent
    burst_test_file = current_directory / "mag_single_norm.pkts"
    l0 = decom_packets(str(burst_test_file))[0]

    assert l0.ccsds_header.PKT_APID == 1052
    assert l0.ccsds_header.SRC_SEQ_CTR == 16
    assert l0.ccsds_header.PKT_LEN == 221

    expected = dataclasses.replace(
        l0,
        SHCOARSE=435954831,
        PUS_STYPE=21,
        PUS_SSUBTYPE=7,
        COMPRESSION=0,
        MAGO_ACT=1,
        MAGI_ACT=1,
        PRI_SENS=0,
        PRI_VECSEC=1,
        SEC_VECSEC=1,
        PRI_COARSETM=435954821,
        PRI_FNTM=64581,
        SEC_COARSETM=435954821,
        SEC_FNTM=64582,
    )
    assert l0 == expected


def test_mag_decom():
    """Testing multiple packets"""
    current_directory = Path(__file__).parent
    burst_test_file = current_directory / "mag_multiple_packets.pkts"
    l0 = decom_packets(str(burst_test_file))

    assert len(l0) == 125
    for val in l0:
        assert val.ccsds_header.PKT_APID == 1052
        assert len(val.VECTORS) > 0

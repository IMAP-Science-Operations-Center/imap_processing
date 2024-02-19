import bitstring
import pandas as pd

from imap_processing import decom, imap_module_directory
from imap_processing.hit.l0.data_classes.message_log import MessageLog


def test_message_log():
    test_file = imap_module_directory / "tests/hit/test_data/msglog_sample.ccsds"
    validation_file = (
        imap_module_directory / "tests/hit/validation_data/msglog_sample_raw.csv"
    )
    xtce_file = imap_module_directory / "hit/packet_definitions/P_HIT_MSGLOG.xml"

    validation_data = pd.read_csv(validation_file)
    packets = decom.decom_packets(test_file.resolve(), xtce_file.resolve())

    for pkt_idx, packet in enumerate(packets):
        msg_log = MessageLog(packet, "0.0", "mslog_sample.ccsds")
        assert msg_log.SHCOARSE == validation_data["SHCOARSE"][pkt_idx]
        assert msg_log.TEXT == bitstring.Bits(hex=validation_data["TEXT"][pkt_idx]).bin

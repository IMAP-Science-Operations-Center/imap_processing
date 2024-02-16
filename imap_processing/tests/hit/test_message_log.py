import pandas as pd

from imap_processing import decom, imap_module_directory


def test_message_log():
    test_file = imap_module_directory / "tests/hit/test_data/msglog_sample.ccsds"
    validation_file = (
        imap_module_directory / "tests/hit/validation_data/msglog_sample_raw.csv"
    )
    xtce_file = imap_module_directory / "hit/packet_definitions/P_HIT_MSGLOG.xml"

    #validation_data = pd.read_csv(validation_file)
    packets = decom.decom_packets(test_file.resolve(), xtce_file.resolve())

    for pkt_idx, packet in enumerate(packets):
        #print(packet)
        print(packet.data["TEXT"].raw_value)

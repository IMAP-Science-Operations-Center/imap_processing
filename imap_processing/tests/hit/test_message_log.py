import numpy as np
import pandas as pd

from imap_processing import decom, imap_module_directory
from imap_processing.hit.l0.data_classes.message_log import MessageLog


def test_houskeeping():
    # The HK validation data's first row was not in the CCSDS file sent, so that has
    # been manually removed from the validation file on the SDC end. The last
    # packet in the CCSDS file also does not correspond to any of the rows in
    # the validation file, so the last packet in the CCSDS file is removed for
    # this test. These issues were discovered / confirmed with the HIT Ops
    # engineer who delivered the data.
    test_file = imap_module_directory / "tests/hit/test_data/msglog_sample.ccsds"
    validation_file = (
        imap_module_directory / "tests/hit/validation_data/msglog_sample_raw.csv"
    )
    xtce_file = imap_module_directory / "hit/packet_definitions/P_HIT_MSGLOG.xml"

    validation_data = pd.read_csv(validation_file)
    packets = decom.decom_packets(test_file.resolve(), xtce_file.resolve())


    for pkt_idx, packet in enumerate(packets):
        print(packet)

import numpy as np
import pandas as pd

from imap_processing import decom, imap_module_directory
from imap_processing.hit.l0.data_classes.housekeeping import Housekeeping


def test_houskeeping():
    # The HK validation data's first row was not in the CCSDS file sent, so that has
    # been manually removed from the validation file on the SDC end. The last
    # packet in the CCSDS file also does not correspond to any of the rows in
    # the validation file, so the last packet in the CCSDS file is removed for
    # this test. These issues were discovered / confirmed with the HIT Ops
    # engineer who delivered the data.
    test_file = imap_module_directory / "tests/hit/test_data/hskp_sample.ccsds"
    validation_file = (
        imap_module_directory / "tests/hit/validation_data/hskp_sample_raw.csv"
    )
    xtce_file = imap_module_directory / "hit/packet_definitions/P_HIT_HSKP.xml"

    validation_data = pd.read_csv(validation_file)
    leak_columns = [col for col in validation_data.columns if col.startswith("LEAK")]
    packets = decom.decom_packets(test_file.resolve(), xtce_file.resolve())[:-1]

    hk_fields = [
        "SHCOARSE",
        "MODE",
        "FSW_VERSION_A",
        "FSW_VERSION_B",
        "FSW_VERSION_C",
        "NUM_GOOD_CMDS",
        "LAST_GOOD_CMD",
        "LAST_GOOD_SEQ_NUM",
        "NUM_BAD_CMDS",
        "LAST_BAD_CMD",
        "LAST_BAD_SEQ_NUM",
        "FEE_RUNNING",
        "MRAM_DISABLED",
        "ENABLE_50KHZ",
        "ENABLE_HVPS",
        "TABLE_STATUS",
        "HEATER_CONTROL",
        "HEATER_CONTROL",
        "ADC_MODE",
        "DYN_THRESH_LVL",
        "NUM_EVNT_LAST_HK",
        "NUM_ERRORS",
        "LAST_ERROR_NUM",
        "CODE_CHECKSUM",
        "SPIN_PERIOD_SHORT",
        "SPIN_PERIOD_LONG",
        "PHASIC_STAT",
        "ACTIVE_HEATER",
        "HEATER_ON",
        "TEST_PULSER_ON",
        "DAC0_ENABLE",
        "DAC1_ENABLE",
        "PREAMP_L234A",
        "PREAMP_L1A",
        "PREAMP_L1B",
        "PREAMP_L234B",
        "TEMP0",
        "TEMP1",
        "TEMP2",
        "TEMP3",
        "ANALOG_TEMP",
        "HVPS_TEMP",
        "IDPU_TEMP",
        "LVPS_TEMP",
        "EBOX_3D4VD",
        "EBOX_5D1VD",
        "EBOX_P12VA",
        "EBOX_M12VA",
        "EBOX_P5D7VA",
        "EBOX_M5D7VA",
        "REF_P5V",
        "L1AB_BIAS",
        "L2AB_BIAS",
        "L34A_BIAS",
        "L34B_BIAS",
        "EBOX_P2D0VD",
    ]

    for pkt_idx, packet in enumerate(packets):
        hk = Housekeeping(packet, "0.0", "hskp_sample.ccsds")
        for field in hk_fields:
            assert getattr(hk, field) == validation_data[field][pkt_idx]

    np.testing.assert_array_equal(
        hk.LEAK_I, np.array(validation_data.loc[pkt_idx, leak_columns].tolist())
    )

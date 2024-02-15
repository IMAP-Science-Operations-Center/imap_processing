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

    for pkt_idx, packet in enumerate(packets):
        hk = Housekeeping(packet, "0.0", "hskp_sample.ccsds")
        assert hk.SHCOARSE == validation_data["SC_TICK"][pkt_idx]
        assert hk.MODE == validation_data["MODE"][pkt_idx]
        assert hk.FSW_VERSION_A == validation_data["FSW_VERSION_A"][pkt_idx]
        assert hk.FSW_VERSION_B == validation_data["FSW_VERSION_B"][pkt_idx]
        assert hk.FSW_VERSION_C == validation_data["FSW_VERSION_C"][pkt_idx]
        assert hk.NUM_GOOD_CMDS == validation_data["NUM_GOOD_CMDS"][pkt_idx]
        assert hk.LAST_GOOD_CMD == validation_data["LAST_GOOD_CMD"][pkt_idx]
        assert hk.LAST_GOOD_SEQ_NUM == validation_data["LAST_GOOD_SEQ_NUM"][pkt_idx]
        assert hk.NUM_BAD_CMDS == validation_data["NUM_BAD_CMDS"][pkt_idx]
        assert hk.LAST_BAD_CMD == validation_data["LAST_BAD_CMD"][pkt_idx]
        assert hk.LAST_BAD_SEQ_NUM == validation_data["LAST_BAD_SEQ_NUM"][pkt_idx]
        assert hk.FEE_RUNNING == validation_data["FEE_RUNNING"][pkt_idx]
        assert hk.MRAM_DISABLED == validation_data["MRAM_DISABLED"][pkt_idx]
        assert hk.ENABLE_50KHZ == validation_data["ENABLE_50KHZ"][pkt_idx]
        assert hk.ENABLE_HVPS == validation_data["ENABLE_HVPS"][pkt_idx]
        assert hk.TABLE_STATUS == validation_data["TABLE_STATUS"][pkt_idx]
        assert hk.HEATER_CONTROL == validation_data["HEATER_CONTROL"][pkt_idx]
        assert hk.ADC_MODE == validation_data["ADC_MODE"][pkt_idx]
        assert hk.DYN_THRESH_LVL == validation_data["DYN_THRESH_LVL"][pkt_idx]
        assert hk.NUM_EVNT_LAST_HK == validation_data["NUM_EVNT_LAST_HK"][pkt_idx]
        assert hk.NUM_ERRORS == validation_data["NUM_ERRORS"][pkt_idx]
        assert hk.LAST_ERROR_NUM == validation_data["LAST_ERROR_NUM"][pkt_idx]
        assert hk.CODE_CHECKSUM == validation_data["CODE_CHECKSUM"][pkt_idx]
        assert hk.SPIN_PERIOD_SHORT == validation_data["SPIN_PERIOD_SHORT"][pkt_idx]
        assert hk.SPIN_PERIOD_LONG == validation_data["SPIN_PERIOD_LONG"][pkt_idx]
        assert (
            hk.LEAK_I == np.array(validation_data.loc[pkt_idx, leak_columns].tolist())
        ).all()
        assert hk.PHASIC_STAT == validation_data["PHASIC_STAT"][pkt_idx]
        assert hk.ACTIVE_HEATER == validation_data["ACTIVE_HEATER"][pkt_idx]
        assert hk.HEATER_ON == validation_data["HEATER_ON"][pkt_idx]
        assert hk.TEST_PULSER_ON == validation_data["TEST_PULSER_ON"][pkt_idx]
        assert hk.DAC0_ENABLE == validation_data["DAC0_ENABLE"][pkt_idx]
        assert hk.DAC1_ENABLE == validation_data["DAC1_ENABLE"][pkt_idx]
        assert hk.PREAMP_L234A == validation_data["PREAMP_L234A"][pkt_idx]
        assert hk.PREAMP_L1A == validation_data["PREAMP_L1A"][pkt_idx]
        assert hk.PREAMP_L1B == validation_data["PREAMP_L1B"][pkt_idx]
        assert hk.PREAMP_L234B == validation_data["PREAMP_L234B"][pkt_idx]
        assert hk.TEMP0 == validation_data["TEMP0"][pkt_idx]
        assert hk.TEMP1 == validation_data["TEMP1"][pkt_idx]
        assert hk.TEMP2 == validation_data["TEMP2"][pkt_idx]
        assert hk.TEMP3 == validation_data["TEMP3"][pkt_idx]
        assert hk.ANALOG_TEMP == validation_data["ANALOG_TEMP"][pkt_idx]
        assert hk.HVPS_TEMP == validation_data["HVPS_TEMP"][pkt_idx]
        assert hk.IDPU_TEMP == validation_data["IDPU_TEMP"][pkt_idx]
        assert hk.LVPS_TEMP == validation_data["LVPS_TEMP"][pkt_idx]
        assert hk.EBOX_3D4VD == validation_data["EBOX_3D4VD"][pkt_idx]
        assert hk.EBOX_5D1VD == validation_data["EBOX_5D1VD"][pkt_idx]
        assert hk.EBOX_P12VA == validation_data["EBOX_P12VA"][pkt_idx]
        assert hk.EBOX_M12VA == validation_data["EBOX_M12VA"][pkt_idx]
        assert hk.EBOX_P5D7VA == validation_data["EBOX_P5D7VA"][pkt_idx]
        assert hk.EBOX_M5D7VA == validation_data["EBOX_M5D7VA"][pkt_idx]
        assert hk.REF_P5V == validation_data["REF_P5V"][pkt_idx]
        assert hk.L1AB_BIAS == validation_data["L1AB_BIAS"][pkt_idx]
        assert hk.L2AB_BIAS == validation_data["L2AB_BIAS"][pkt_idx]
        assert hk.L34A_BIAS == validation_data["L34A_BIAS"][pkt_idx]
        assert hk.L34B_BIAS == validation_data["L34B_BIAS"][pkt_idx]
        assert hk.EBOX_P2D0VD == validation_data["EBOX_P2D0VD"][pkt_idx]

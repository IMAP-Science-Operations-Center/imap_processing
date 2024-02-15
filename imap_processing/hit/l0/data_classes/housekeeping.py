"""L1A HIT Housekeeping data class."""
from dataclasses import dataclass
import bitstring
import numpy as np
from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.hit.l0.hit_base import HITBase

@dataclass
class Housekeeping(HITBase):

    SHCOARSE: int
    MODE: int
    FSW_VERSION_A: int
    FSW_VERSION_B: int
    FSW_VERSION_C: int
    NUM_GOOD_CMDS: int
    LAST_GOOD_CMD: int
    LAST_GOOD_SEQ_NUM: int
    NUM_BAD_CMDS: int
    LAST_BAD_CMD: int
    LAST_BAD_SEQ_NUM: int
    FEE_RUNNING: int
    MRAM_DISABLED: int
    ENABLE_50KHZ: int
    ENABLE_HVPS: int
    TABLE_STATUS: int
    HEATER_CONTROL: int
    ADC_MODE: int
    DYN_THRESH_LVL: int
    NUM_EVNT_LAST_HK: int
    NUM_ERRORS: int
    LAST_ERROR_NUM: int
    CODE_CHECKSUM: int
    SPIN_PERIOD_SHORT: int
    SPIN_PERIOD_LONG: int
    LEAK_I_RAW: str
    LEAK_I: np.ndarray
    PHASIC_STAT: int
    ACTIVE_HEATER: int
    HEATER_ON: int
    TEST_PULSER_ON: int
    DAC0_ENABLE: int
    DAC1_ENABLE: int
    PREAMP_L234A: int
    PREAMP_L1A: int
    PREAMP_L1B: int
    PREAMP_L234B: int
    TEMP0: int
    TEMP1: int
    TEMP2: int
    TEMP3: int
    ANALOG_TEMP: int
    HVPS_TEMP: int
    IDPU_TEMP: int
    LVPS_TEMP: int
    EBOX_3D4VD: int
    EBOX_5D1VD: int
    EBOX_P12VA: int
    EBOX_M12VA: int
    EBOX_P5D7VA: int
    EBOX_M5D7VA: int
    REF_P5V: int
    L1AB_BIAS: int
    L2AB_BIAS: int
    L34A_BIAS: int
    L34B_BIAS: int
    EBOX_P2D0VD: int

    def __init__(self, packet, software_version: str, packet_file_name: str):
        """Intialization method for Housekeeping Data class."""
        super().__init__(software_version, packet_file_name, CcsdsData(packet.header))
        self.parse_data(packet)
        self._parse_leak()
    
    def _parse_leak(self):
        leak_i_list = list()
        leak_bits = bitstring.Bits(bin=self.LEAK_I_RAW)
        index_bit_length = 10
        for leak_idx in range(640, 0, -index_bit_length):
            leak_i_list.append(leak_bits[leak_idx - index_bit_length:leak_idx].uint)
        self.LEAK_I = np.array(leak_i_list)


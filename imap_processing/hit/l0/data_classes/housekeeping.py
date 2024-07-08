"""L1A HIT Housekeeping data class."""

from dataclasses import dataclass

import numpy as np
import space_packet_parser

from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.hit.l0.utils.hit_base import HITBase


@dataclass
class Housekeeping(HITBase):
    """
    L1A HIT Housekeeping data.

    The HIT Housekeeping data class handles the decommutation
    and parsing of L0 to L1A data.

    Parameters
    ----------
    packet : dict
        Dictionary of packet.
    software_version : str
        Version of software.
    packet_file_name : str
        Name of packet file.

    Attributes
    ----------
    SHCOARSE : int
        Spacecraft time.
    MODE : int
        Mode (0=boot, 1=maint, 2=stdby, 3=science)
    FSW_VERSION_A : int
        FSW version number (A.B.C bits)
    FSW_VERSION_B : int
        FSW version number (A.B.C bits)
    FSW_VERSION_C : int
        FSW version number (A.B.C bits)
    NUM_GOOD_CMDS : int
        Number of good commands
    LAST_GOOD_CMD : int
        Last good command
    LAST_GOOD_SEQ_NUM : int
        Last good sequence number
    NUM_BAD_CMDS : int
        Number of bad commands
    LAST_BAD_CMD : int
        Last bad command
    LAST_BAD_SEQ_NUM : int
        Last bad sequence number
    FEE_RUNNING : int
        FEE running (1) or reset (0)
    MRAM_DISABLED : int
        MRAM disabled (1) or enabled (0)
    ENABLE_50KHZ : int
        50kHz enabled (1) or disabled (0)
    ENABLE_HVPS : int
        HVPS enabled (1) or disabled (0)
    TABLE_STATUS : int
        Table status  OK (1) or error (0)
    HEATER_CONTROL : int
        Heater control (0=none, 1=pri, 2=sec)
    ADC_MODE : int
        ADC mode (0=quiet, 1=normal, 2=adcstim, 3=adcThreshold?)
    DYN_THRESH_LVL : int
        Dynamic threshold level (0-3)
    NUM_EVNT_LAST_HK : int
        Number of events since last HK update
    NUM_ERRORS : int
        Number of errors
    LAST_ERROR_NUM : int
        Last error number
    CODE_CHECKSUM : int
        Code checksum
    SPIN_PERIOD_SHORT : int
        Spin period at t=0
    SPIN_PERIOD_LONG : int
        Spin period at t=0
    LEAK_I_RAW : str
        Raw binary for Leakage current [V]
    LEAK_I : np.ndarray
        Leakage currents [V] formatted as (64, 1) array
    PHASIC_STAT : int
        PHASIC status
    ACTIVE_HEATER : int
        Active heater
    HEATER_ON : int
        Heater on/off
    TEST_PULSER_ON : int
        Test pulser on/off
    DAC0_ENABLE : int
        DAC_0 enable
    DAC1_ENABLE : int
        DAC_1 enable
    PREAMP_L234A : int
        Preamp L234A
    PREAMP_L1A : int
        Preamp L1A
    PREAMP_L1B : int
        Preamp L1B
    PREAMP_L234B : int
        Preamp L234B
    TEMP0 : int
        FEE LDO Regulator
    TEMP1 : int
        Primary Heater
    TEMP2 : int
        FEE FPGA
    TEMP3 : int
        Secondary Heater
    ANALOG_TEMP : int
        Chassis temp
    HVPS_TEMP : int
        Board temp
    IDPU_TEMP : int
        LDO Temp
    LVPS_TEMP : int
        Board temp
    EBOX_3D4VD : int
        3.4VD Ebox (digital)
    EBOX_5D1VD : int
        5.1VD Ebox (digital)
    EBOX_P12VA : int
        +12VA Ebox (analog)
    EBOX_M12VA : int
        -12VA Ebox (analog)
    EBOX_P5D7VA : int
        +5.7VA Ebox (analog)
    EBOX_M5D7VA : int
        -5.7VA Ebox (analog)
    REF_P5V : int
        +5Vref
    L1AB_BIAS : int
        L1A/B Bias
    L2AB_BIAS : int
        L2A/B Bias
    L34A_BIAS : int
        L3/4A Bias
    L34B_BIAS : int
        L3/4B Bias
    EBOX_P2D0VD : int
        +2.0VD Ebox (digital)

    Methods
    -------
    __init__(packet, software_vesion, packet_file_name):
        Uses the CCSDS packet, version of the software, and
        the name of the packet file to parse and store information about
        the Housekeeping packet data.
    _parse_leak():
        Parse each current leakage field and put into an array.
    """

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

    def __init__(
        self,
        packet: space_packet_parser.parser.Packet,
        software_version: str,
        packet_file_name: str,
    ):
        """Housekeeping Data class initialization method."""
        super().__init__(software_version, packet_file_name, CcsdsData(packet.header))
        self.parse_data(packet)
        self._parse_leak()

    def _parse_leak(self) -> None:
        """Parse each current leakage field and put into an array."""
        # Each Leak field is 10 bits long
        leak_bit_length = 10
        # There are 64 leak fields
        num_leak_fields = 64
        self.LEAK_I = np.empty(num_leak_fields, dtype=np.uint16)
        # The leak fields appear in the packet in ascending order, so to append
        # the leak fields in the correct order, the binary will be parsed
        # from right to left.
        for i, leak_idx in enumerate(
            range(leak_bit_length * num_leak_fields, 0, -leak_bit_length)
        ):
            self.LEAK_I[i] = int(
                self.LEAK_I_RAW[leak_idx - leak_bit_length : leak_idx], 2
            )

import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture(scope="session")
def xtce_mag_path():
    """Returns the xtce auxiliary directory."""
    return imap_module_directory / "ialirt" / "packet_definitions" / "ialirt_mag.xml"


@pytest.fixture(scope="session")
def binary_packet_path():
    """Returns the xtce directory."""
    return (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "test_data"
        / "l0"
        / "461971314-335.bin"
    )


@pytest.fixture(scope="session")
def mag_test_data():
    """Returns the test data directory."""
    data_path = (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "test_data"
        / "l0"
        / "sample decoded i-alirt data.csv"
    )
    data = pd.read_csv(data_path)

    return data


@pytest.fixture()
def xarray_data(binary_packet_path, xtce_mag_path):
    """Create xarray data"""
    apid = 1001

    xarray_data = packet_file_to_datasets(
        binary_packet_path, xtce_mag_path, use_derived_value=False
    )[apid]

    return xarray_data


def test_decom_packets(xarray_data, mag_test_data):
    """This function checks that all instrument parameters are accounted for."""

    print('hi')


import Gseos
import GseosBlocks
import GseosDecoder
import csv
from datetime import datetime
import time
from GseosConversion import *

PRINT_GAPS = False


class IAlirtPacket:
    def __init__(self, blk):

        blk_data = blk.ReadBlockAsBytes()

        # parse the I-ALiRT data
        self.sequence = getattr(blk, 'PHSEQCNT')
        self.status = ((blk_data[16] << 16) | \
                       (blk_data[17] << 8) | \
                       (blk_data[18] << 0)) & 0xFFFFFF
        self.scienceData = blk_data[19:22]

        # Get packet sequence
        self.pkt_counter = (blk_data[16] >> 6) & 0x03

        # pull out HK bits
        if self.pkt_counter == 0:
            self.HK1V5_WARN = ((blk_data[16] >> 4) & 0x01)
            self.HK1V5_DANGER = ((blk_data[16] >> 3) & 0x01)
            self.HK1V5C_WARN = ((blk_data[16] >> 2) & 0x01)
            self.HK1V5C_DANGER = ((blk_data[16] >> 1) & 0x01)
            self.HK1V8_WARN = ((blk_data[16] >> 0) & 0x01)
            self.HK1V8_DANGER = ((blk_data[17] >> 7) & 0x01)
            self.HK1V8C_WARN = ((blk_data[17] >> 6) & 0x01)
            self.HK1V8C_DANGER = ((blk_data[17] >> 5) & 0x01)
            self.FOB_SATURATED = ((blk_data[18] >> 5) & 0x01)
            self.FIB_SATURATED = ((blk_data[18] >> 4) & 0x01)
            self.MODE = ((blk_data[18] >> 0) & 0xF)
            self.ICU_TEMP = ((self.status >> 6) & 0x7F) << 5

            self.science0 = self.scienceData

        elif self.pkt_counter == 1:
            self.HK2V5_WARN = ((blk_data[16] >> 4) & 0x01)
            self.HK2V5_DANGER = ((blk_data[16] >> 3) & 0x01)
            self.HK2V5C_WARN = ((blk_data[16] >> 2) & 0x01)
            self.HK2V5C_DANGER = ((blk_data[16] >> 1) & 0x01)
            self.HK3V3 = ((self.status >> 9) & 0xFF) << 4
            self.HK3V3_CURRENT = ((self.status >> 0) & 0x1FF) << 3
            self.PRI_COARSETM = ((blk_data[10] << 24) | \
                                 (blk_data[11] << 16) | \
                                 (blk_data[12] << 8) | \
                                 (blk_data[13] << 0)) & 0xFFFFFFFF
            self.PRI_FINTM = ((blk_data[14] << 8) | \
                              (blk_data[15] << 0)) & 0xFFFF
            self.pri_isValid = ((blk_data[16] >> 5) & 0x01)

            self.science1 = self.scienceData

        elif self.pkt_counter == 2:
            self.HKP8V5_WARN = ((blk_data[16] >> 4) & 0x01)
            self.HKP8V5_DANGER = ((blk_data[16] >> 3) & 0x01)
            self.HKP8V5C_WARN = ((blk_data[16] >> 2) & 0x01)
            self.HKP8V5C_DANGER = ((blk_data[16] >> 1) & 0x01)
            self.HKN8V5 = ((self.status >> 9) & 0xFF) << 4
            self.HKN8V5_CURRENT = ((self.status >> 0) & 0x1FF) << 3

            self.science2 = self.scienceData


        elif self.pkt_counter == 3:
            self.FOB_TEMP = ((self.status >> 13) & 0xFF) << 4
            self.FIB_TEMP = ((self.status >> 5) & 0xFF) << 4
            self.FOB_RANGE = ((self.status >> 3) & 0x3)
            self.FIB_RANGE = ((self.status >> 1) & 0x3)
            self.MULTBIT_ERRS = ((self.status >> 0) & 0x1)
            self.SEC_COARSETM = ((blk_data[10] << 24) | \
                                 (blk_data[11] << 16) | \
                                 (blk_data[12] << 8) | \
                                 (blk_data[13] << 0)) & 0xFFFFFFFF
            self.SEC_FINTM = ((blk_data[14] << 8) | \
                              (blk_data[15] << 0)) & 0xFFFF
            self.sec_isValid = ((blk_data[16] >> 5) & 0x01)

            self.science3 = self.scienceData

            # Concatenate Science data
            self.priX = toSigned16(self.science0[0] << 8 | self.science0[1])
            self.priY = toSigned16(self.science0[2] << 8 | self.science1[0])
            self.priZ = toSigned16(self.science1[1] << 8 | self.science1[2])

            self.secX = toSigned16(self.science2[0] << 8 | self.science2[1])
            self.secY = toSigned16(self.science2[2] << 8 | self.science3[0])
            self.secZ = toSigned16(self.science3[1] << 8 | self.science3[2])

        self.newBlock = self.writeBlock(blk)

    def zero(self):
        self.pri_isValid = 0
        self.sec_isValid = 0
        self.PRI_COARSETM = 0
        self.PRI_FINTM = 0
        self.SEC_COARSETM = 0
        self.SEC_FINTM = 0
        self.HK1V5_WARN = 0
        self.HK1V5_DANGER = 0
        self.HK1V5C_WARN = 0
        self.HK1V5C_DANGER = 0
        self.HK1V8_WARN = 0
        self.HK1V8_DANGER = 0
        self.HK1V8C_WARN = 0
        self.HK1V8C_DANGER = 0
        self.FOB_SATURATED = 0
        self.FIB_SATURATED = 0
        self.MODE = 0
        self.ICU_TEMP = 0
        self.HK2V5_WARN = 0
        self.HK2V5_DANGER = 0
        self.HK2V5C_WARN = 0
        self.HK2V5C_DANGER = 0
        self.HK3V3 = 0
        self.HK3V3_CURRENT = 0
        self.HKP8V5_WARN = 0
        self.HKP8V5_DANGER = 0
        self.HKP8V5C_WARN = 0
        self.HKP8V5C_DANGER = 0
        self.HKN8V5 = 0
        self.HKN8V5_CURRENT = 0
        self.FOB_TEMP = 0
        self.FIB_TEMP = 0
        self.FOB_RANGE = 0
        self.FIB_RANGE = 0
        self.MULTBIT_ERRS = 0
        self.priX = 0
        self.priY = 0
        self.priZ = 0
        self.secX = 0
        self.secY = 0
        self.secZ = 0
        self.sequence = 0

    def writeBlock(self, blk):
        global ptime

        if self.pkt_counter != 3:
            pass
        elif self.pkt_counter == 3:
            plotBlk_ialirt.PRI_ISVALID = self.pri_isValid
            plotBlk_ialirt.SEC_ISVALID = self.sec_isValid
            plotBlk_ialirt.PRI_COARSETM = self.PRI_COARSETM
            plotBlk_ialirt.PRI_FINTM = self.PRI_FINTM
            plotBlk_ialirt.SEC_COARSETM = self.SEC_COARSETM
            plotBlk_ialirt.SEC_FINTM = self.SEC_FINTM
            plotBlk_ialirt.HK1V5_WARN = self.HK1V5_WARN
            plotBlk_ialirt.HK1V5_DANGER = self.HK1V5_DANGER
            plotBlk_ialirt.HK1V5C_WARN = self.HK1V5C_WARN
            plotBlk_ialirt.HK1V5C_DANGER = self.HK1V5C_DANGER
            plotBlk_ialirt.HK1V8_WARN = self.HK1V8_WARN
            plotBlk_ialirt.HK1V8_DANGER = self.HK1V8_DANGER
            plotBlk_ialirt.HK1V8C_WARN = self.HK1V8C_WARN
            plotBlk_ialirt.HK1V8C_DANGER = self.HK1V8C_DANGER
            plotBlk_ialirt.FOB_SATURATED = self.FOB_SATURATED
            plotBlk_ialirt.FIB_SATURATED = self.FIB_SATURATED
            plotBlk_ialirt.MODE = self.MODE
            plotBlk_ialirt.ICU_TEMP = self.ICU_TEMP
            plotBlk_ialirt.HK2V5_WARN = self.HK2V5_WARN
            plotBlk_ialirt.HK2V5_DANGER = self.HK2V5_DANGER
            plotBlk_ialirt.HK2V5C_WARN = self.HK2V5C_WARN
            plotBlk_ialirt.HK2V5C_DANGER = self.HK2V5C_DANGER
            plotBlk_ialirt.HK3V3 = self.HK3V3
            plotBlk_ialirt.HK3V3_CURRENT = self.HK3V3_CURRENT
            plotBlk_ialirt.HKP8V5_WARN = self.HKP8V5_WARN
            plotBlk_ialirt.HKP8V5_DANGER = self.HKP8V5_DANGER
            plotBlk_ialirt.HKP8V5C_WARN = self.HKP8V5C_WARN
            plotBlk_ialirt.HKP8V5C_DANGER = self.HKP8V5C_DANGER
            plotBlk_ialirt.HKN8V5 = self.HKN8V5
            plotBlk_ialirt.HKN8V5_CURRENT = self.HKN8V5_CURRENT
            plotBlk_ialirt.FOB_TEMP = self.FOB_TEMP
            plotBlk_ialirt.FIB_TEMP = self.FIB_TEMP
            plotBlk_ialirt.FOB_RANGE = self.FOB_RANGE
            plotBlk_ialirt.FIB_RANGE = self.FIB_RANGE
            plotBlk_ialirt.MULTBIT_ERRS = self.MULTBIT_ERRS
            plotBlk_ialirt.PRI_X = self.priX
            plotBlk_ialirt.PRI_Y = self.priY
            plotBlk_ialirt.PRI_Z = self.priZ
            plotBlk_ialirt.SEC_X = self.secX
            plotBlk_ialirt.SEC_Y = self.secY
            plotBlk_ialirt.SEC_Z = self.secZ
            plotBlk_ialirt.SEQUENCE = self.sequence

            if self.pri_isValid != 0 or self.sec_isValid != 0 and PRINT_GAPS:
                primaryTime = float(self.PRI_COARSETM) + (float(self.PRI_FINTM) / 65536)
                secondaryTime = float(self.SEC_COARSETM) + (float(self.SEC_FINTM) / 65536)
                diffInMs = ((primaryTime - secondaryTime) * 1000)
                print(
                    f"FIBFOB I-ALiRT +{primaryTime - ptime:.1f} diff is {diffInMs:.1f}ms. Primary time is {primaryTime:.4f} and secondary time is {secondaryTime:.4f}")
                ptime = primaryTime

            plotBlk_ialirt.SendBlock()
            self.zero()


def toSigned16(n):
    n = n & 0xffff
    return n | (-(n & 0x8000))


# on setup of decoder, grab the blocks and attach the decoder
plotBlk_ialirt = GseosBlocks.Blocks['MAG_IALIRTHK']
sourceIalirt_blk = GseosBlocks.Blocks['MAG_SCI_IALIRT']

packet = IAlirtPacket(sourceIalirt_blk)
ptime = float(0)

oIalirtDecoder = GseosDecoder.TDecoder('I-ALiRT HK Decoder', packet.__init__, [plotBlk_ialirt])
sourceIalirt_blk.Decoders.append(oIalirtDecoder)



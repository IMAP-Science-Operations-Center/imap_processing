"""
Various classes and functions used throughout CoDICE processing.

This module contains utility classes and functions that are used by various
other CoDICE processing modules.
"""

from enum import IntEnum

import numpy as np
import pandas as pd


class CODICEAPID(IntEnum):
    """Create ENUM for CoDICE APIDs."""

    COD_AUT = 1120
    COD_BOOT_HK = 1121
    COD_BOOT_MEMDMP = 1122
    COD_COUNTS_COMMON = 1135
    COD_NHK = 1136
    COD_EVTMSG = 1137
    COD_MEMDMP = 1138
    COD_SHK = 1139
    COD_RTS = 1141
    COD_DIAG_CDHFPGA = 1144
    COD_DIAG_SNSR_HV = 1145
    COD_DIAG_OPTC_HV = 1146
    COD_DIAG_APDFPGA = 1147
    COD_DIAG_SSDFPGA = 1148
    COD_DIAG_FSW = 1149
    COD_DIAG_SYSVARS = 1150
    COD_LO_IAL = 1152
    COD_LO_PHA = 1153
    COD_LO_SW_PRIORITY_COUNTS = 1155
    COD_LO_SW_SPECIES_COUNTS = 1156
    COD_LO_NSW_SPECIES_COUNTS = 1157
    COD_LO_SW_ANGULAR_COUNTS = 1158
    COD_LO_NSW_ANGULAR_COUNTS = 1159
    COD_LO_NSW_PRIORITY_COUNTS = 1160
    COD_LO_INST_COUNTS_AGGREGATED = 1161
    COD_LO_INST_COUNTS_SINGLES = 1162
    COD_HI_IAL = 1168
    COD_HI_PHA = 1169
    COD_HI_INST_COUNTS_AGGREGATED = 1170
    COD_HI_INST_COUNTS_SINGLES = 1171
    COD_HI_OMNI_SPECIES_COUNTS = 1172
    COD_HI_SECT_SPECIES_COUNTS = 1173
    COD_HI_INST_COUNTS_PRIORITIES = 1174
    COD_CSTOL_CONFIG = 2457


class CoDICECompression(IntEnum):
    """Create ENUM for CoDICE compression algorithms."""

    NO_COMPRESSION = 0
    LOSSY_A = 1
    LOSSY_B = 2
    LOSSLESS = 3
    LOSSY_A_LOSSLESS = 4
    LOSSY_B_LOSSLESS = 5
    PACK_24_BIT = 6


def reshape_ssd_energy_df(ssd_energy_df: pd.DataFrame) -> np.ndarray:
    """
    Reshape the SSD energy dataframe into a 3D array.

    Data frame has data in this format:
    bin_num,SSD 0 - LG ,SSD 0 - MG,SSD 0 - HG, ...,SSD 15 - LG ,SSD 15 - MG,SSD 15 - HG

    Now we want to reformat the data into this resulting array will have
    dimensions (rows, ssd, gain), where:
        - rows is the number of bin values in the dataframe
        - ssd is the number of SSDs (0-15)
        - gain is the number of gain modes (No Gain, LG, MG, HG)

    Parameters
    ----------
    ssd_energy_df : pandas.DataFrame
        The SSD energy dataframe with columns like 'SSD 0 - LG', 'SSD 0 - MG', etc.

    Returns
    -------
    numpy.ndarray
        A 3D array with dimensions (rows, ssd, gain).
    """
    num_ssds = 16
    num_gains = 4  # [No, LG, MG, HG]

    # Extract just the SSD energy columns (exclude bin_num etc.)
    energy_values = ssd_energy_df.drop(columns=["bin_num"], errors="ignore").to_numpy()

    # Reshape into (rows, ssds, gains-1)
    arr = energy_values.reshape(len(ssd_energy_df), num_ssds, num_gains - 1)

    # Prepend the "No energy" gain (all NaNs)
    no_energy = np.full((len(ssd_energy_df), num_ssds, 1), np.nan)

    # Concatenate to get (rows, ssds, 4 gains)
    ssd_energy_3d = np.concatenate([no_energy, arr], axis=-1)

    return ssd_energy_3d

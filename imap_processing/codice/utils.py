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

    The resulting array will have dimensions (rows, ssd, gain), where:
    - rows is the number of bin values in the dataframe
    - ssd is the number of SSDs (0-15)
    - gain is the number of gain modes (LG, MG, HG)

    Parameters
    ----------
    ssd_energy_df : pandas.DataFrame
        The SSD energy dataframe with columns like 'SSD 0 - LG', 'SSD 0 - MG', etc.

    Returns
    -------
    numpy.ndarray
        A 3D array with dimensions (rows, ssd, gain).
    """
    # Get the number of rows in the dataframe
    num_rows = len(ssd_energy_df)
    # Number of SSDs (0-15)
    num_ssds = 16
    # Number of gain modes (No Gain, LG, MG, HG)
    num_gains = 4

    # Create an empty 3D array
    ssd_energy_3d = np.full((num_rows, num_ssds, num_gains), np.nan)

    # Map gain mode strings to indices
    # 0: No energy, 1: Low Gain (LG), 2: Mid Gain (MG), 3: High Gain (HG)
    gain_mode_map = {"No energy": 0, "LG": 1, "MG": 2, "HG": 3}

    # Fill the 3D array
    for ssd_id in range(num_ssds):
        for gain_name, gain_idx in gain_mode_map.items():
            col_name = f"SSD {ssd_id} - {gain_name}"
            if col_name in ssd_energy_df.columns:
                ssd_energy_3d[:, ssd_id, gain_idx] = ssd_energy_df[col_name].values

    return ssd_energy_3d

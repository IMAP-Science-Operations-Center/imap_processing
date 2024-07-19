"""Contains tools for lookup tables for l1b."""

import numpy as np
import pandas as pd

from imap_processing import imap_module_directory

base_path = imap_module_directory / "ultra" / "lookup_tables"


def get_y_adjust(dy_lut: np.ndarray):
    """
    Adjust the front yf position based on the particle's trajectory.

    Instead of using trigonometry, this function utilizes a 256-element lookup table
    to find the Y adjustment. For more details, refer to pages 37-38 of the
    IMAP-Ultra Flight Software Specification document (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    dy_lut : np.ndarray
        Change in y direction used for the lookup table (mm).

    Returns
    -------
    yadj : np.ndarray
        Y adjustment (mm).
    """
    yadjust_path = base_path / "yadjust.csv"
    yadjust_df = pd.read_csv(yadjust_path).set_index("dYLUT")

    yadj = yadjust_df["dYAdj"].iloc[dy_lut]

    return yadj.values


def get_norm(dn: np.ndarray, key: str, sensor: str):
    """
    Correct mismatches between the stop Time to Digital Converters (TDCs).

    There are mismatches between the stop TDCs, i.e., SpN, SpS, SpE, and SpW.
    Before these can be used, they must be corrected, or normalized,
    using lookup tables.

    Further description is available on pages 31-32 of the IMAP-Ultra Flight Software
    Specification document (7523-9009_Rev_-.pdf). This will work for both Tp{key}Norm,
    Bt{key}Norm. This is for getStopNorm and getCoinNorm.

    Parameters
    ----------
    dn : np.ndarray
        DN of the TDC.
    key : str
        TpSpNNorm, TpSpSNorm, TpSpENorm, or TpSpWNorm.
        BtSpNNorm, BtSpSNorm, BtSpENorm, or BtSpWNorm.
    sensor : str
        Instrument (ultra45 or ultra90).

    Returns
    -------
    dn_norm : np.ndarray
        Normalized DNs.
    """
    # We only need the center string, i.e. SpN, SpS, SpE, SpW
    if sensor == "ultra45":
        file_label = "Ultra45_tdc_norm_LUT_IntPulser_20230901.csv"
    else:
        file_label = "Ultra90_tdc_norm_LUT_IntPulser_20230614.csv"

    tdc_norm_path = base_path / file_label
    tdc_norm_df = pd.read_csv(tdc_norm_path, header=0, index_col="Index")

    dn_norm = tdc_norm_df[key].iloc[dn]

    return dn_norm.values


def get_back_position(back_index: np.ndarray, key: str, sensor: str):
    """
    Convert normalized TDC values using lookup tables.

    The anodes behave non-linearly near their edges; thus, the use of lookup tables
    instead of linear equations is necessary. The computation will use different
    tables to accommodate variations between the top and bottom anodes.
    Further description is available on page 32 of the
    IMAP-Ultra Flight Software Specification document (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    back_index : np.ndarray
        Options include SpSNorm - SpNNorm + 2047, SpENorm - SpWNorm + 2047,
        SpSNorm - SpNNorm + 2047, or SpENorm - SpWNorm + 2047.
    key : str
        XBkTp, YBkTp, XBkBt, or YBkBt.
    sensor : str
        Instrument (ultra45 or ultra90).

    Returns
    -------
    dn_converted : np.ndarray
        Converted DNs to Units of hundredths of a millimeter.
    """
    if sensor == "ultra45":
        file_label = "back-pos-luts_SN202_20230216.csv"
    else:
        file_label = "back-pos-luts_SN201_20230717.csv"

    back_pos_path = base_path / file_label
    back_pos_df = pd.read_csv(back_pos_path, index_col="Index_offset")

    dn_converted = back_pos_df[key].iloc[back_index]

    return dn_converted.values


def get_energy_norm(ssd: np.ndarray, composite_energy: np.ndarray):
    """
    Normalize composite energy per SSD using a lookup table.

    Further description is available on page 41 of the
    IMAP-Ultra Flight Software Specification document
    (7523-9009_Rev_-.pdf). Note : There are 8 SSDs containing
    4096 composite energies each.

    Parameters
    ----------
    ssd : np.ndarray
        Acts as index 1.
    composite_energy : np.ndarray
        Acts as index 2.

    Returns
    -------
    norm_composite_energy : np.ndarray
        Normalized composite energy.
    """
    energy_norm_path = base_path / "EgyNorm.mem.csv"
    energy_norm_df = pd.read_csv(energy_norm_path)

    row_number = ssd * 4096 + composite_energy
    norm_composite_energy = energy_norm_df["NormEnergy"].iloc[row_number]

    return norm_composite_energy.values


def get_image_params(image: str):
    """
    Lookup table for image parameters.

    Further description is available starting on
    page 30 of the IMAP-Ultra Flight Software
    Specification document (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    image : str
        The column name to lookup in the CSV file, e.g., 'XFTLTOFF' or 'XFTRTOFF'.

    Returns
    -------
    value : np.float64
        Image parameter value from the CSV file.
    """
    csv_file_path = base_path / "FM45_Startup1_ULTRA_IMGPARAMS_20240207T134735_.csv"
    df = pd.read_csv(csv_file_path)
    value = df[image].iloc[0]

    return value

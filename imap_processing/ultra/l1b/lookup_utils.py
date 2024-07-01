"""Contains tools for lookup tables for l1b."""

import pandas as pd

from imap_processing import imap_module_directory

base_path = f"{imap_module_directory}/ultra/lookup_tables"


def get_y_adjust(dy_lut: int):
    """
    Adjust the front yf position based on the particle's trajectory.

    Instead of using trigonometry, this function utilizes a 256-element lookup table
    to find the Y adjustment. For more details, refer to pages 37-38 of the
    IMAP-Ultra Flight Software Specification document (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    dy_lut : int
        Change in y direction used for the lookup table (mm).

    Returns
    -------
    yadj : int
        Y adjustment (mm).
    """
    yadjust_path = f"{base_path}/yadjust.csv"
    yadjust_df = pd.read_csv(yadjust_path).set_index("dYLUT")

    yadj = yadjust_df["dYAdj"].iloc[dy_lut]

    return yadj


def get_norm(dn: int, key: str, file_label: str):
    """
    Correct mismatches between the stop Time to Digital Converters (TDCs).

    There are mismatches between the stop TDCs, i.e., SpN, SpS, SpE, and SpW.
    Before these can be used, they must be corrected, or normalized,
    using lookup tables.

    Further description is available on pages 31-32 of the IMAP-Ultra Flight Software
    Specification document (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    dn : int
        DN of the TDC.
    key : str
        TpSpNNorm, TpSpSNorm, TpSpENorm, or TpSpWNorm.
        BtSpNNorm, BtSpSNorm, BtSpENorm, or BtSpWNorm.
    file_label : str
        Instrument (ultra45 or ultra90).

    Note: This will work for both Tp{key}Norm and Bt{key}Norm
    This is for getStopNorm and getCoinNorm.

    Returns
    -------
    dn_norm : int
        Normalized DNs.
    """
    # We only need the center string, i.e. SpN, SpS, SpE, SpW
    tdc_norm_path = f"{base_path}/{file_label}_tdc_norm.csv"
    tdc_norm_df = pd.read_csv(tdc_norm_path, header=1, index_col="Index")

    dn_norm = tdc_norm_df[key].iloc[dn]

    return dn_norm.values


def get_back_position(back_index: int, key: str, file_label: str):
    """
    Convert normalized TDC values using lookup tables.

    The anodes behave non-linearly near their edges; thus, the use of lookup tables
    instead of linear equations is necessary. The computation will use different
    tables to accommodate variations between the top and bottom anodes.
    Further description is available on page 32 of the
    IMAP-Ultra Flight Software Specification document (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    back_index : int
        dn_norm (output from get_norm).
        Options include SpSNorm - SpNNorm + 2047, SpENorm - SpWNorm + 2047,
        SpSNorm - SpNNorm + 2047, or SpENorm - SpWNorm + 2047
    key : str
        XBkTp, YBkTp, XBkBt, or YBkBt
    file_label : str
        Instrument (ultra45 or ultra90).

    Returns
    -------
    dn_converted : int
        Converted DNs to Units of hundredths of a millimeter.
    """
    back_pos_path = f"{base_path}/{file_label}_back-pos-luts.csv"
    back_pos_df = pd.read_csv(back_pos_path, index_col="Index_offset")

    dn_converted = back_pos_df[key].iloc[back_index]

    return dn_converted


def get_energy_norm(ssd, composite_energy):
    """
    Normalize composite energy per SSD using a lookup table.

    Further description is available on page 41 of the
    IMAP-Ultra Flight Software Specification document
    (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    ssd : int
        Acts as index 1.
    composite_energy : int
        Acts as index 2.

    Note: There are 8 SSDs containing
    4096 composite energies each.

    Returns
    -------
    norm_composite_energy : int
        Normalized composite energy.
    """
    energy_norm_path = f"{base_path}/EgyNorm.mem.csv"
    energy_norm_df = pd.read_csv(energy_norm_path)

    row_number = ssd * 4096 + composite_energy
    norm_composite_energy = energy_norm_df["NormEnergy"].iloc[row_number]

    return norm_composite_energy


def get_image_params(image: str):
    """
    Lookup table for image parameters.

    Further description is available starting on
    page 30 of the IMAP-Ultra Flight Software
    Specification document (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    image : str
        The column name to lookup in the CSV file, e.g., 'XFtLtOff' or 'XFtRtOff'.
    base_path : str
        The base path where the CSV file is stored.

    Returns
    -------
    value : float
        Image parameter value from the CSV file.
    """
    csv_file_path = f"{base_path}/FM45_Startup1_ULTRA_IMGPARAMS_20240207T134735_.csv"
    df = pd.read_csv(csv_file_path)
    value = df[image].iloc[0]

    return value

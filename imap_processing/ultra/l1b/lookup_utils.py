"""Contains tools for lookup tables for l1b."""

import re

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

    if dy_lut < 0:
        dy_lut = 0
    yadj = yadjust_df.at[dy_lut, "dYAdj"]

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
    match = re.search(r"(Sp[NSEW])", key)
    search_key = match.group(1)

    tdc_norm_path = f"{base_path}/{file_label}_tdc_norm.csv"
    tdc_norm_df = pd.read_csv(tdc_norm_path, header=1)

    dn_norm = tdc_norm_df.at[dn, search_key]

    return dn_norm


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

    dn_converted = back_pos_df.at[back_index, key]

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
    norm_composite_energy = energy_norm_df.at[row_number, "NormEnergy"]

    return norm_composite_energy


def get_image_params(image: str):
    """
    Lookup table for image parameters.

    Further description is available starting on
    page 30 of the IMAP-Ultra Flight Software
    Specification document (7523-9009_Rev_-.pdf).

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
    value_sw : int
        Image parameter.
    """
    image_params_path = f"{base_path}/Ultra90_image-params071823.xlsx"
    image_params_df = pd.read_excel(
        image_params_path,
        usecols=["Name", "Value (SW)"],
        index_col="Name",
        engine="openpyxl",
    )
    value_sw = image_params_df.loc[image, "Value (SW)"]

    return value_sw

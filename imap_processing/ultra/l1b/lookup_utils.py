"""Contains tools for lookup tables for l1b."""

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from imap_processing import imap_module_directory

BASE_PATH = imap_module_directory / "ultra" / "lookup_tables"

_YADJUST_DF = pd.read_csv(BASE_PATH / "yadjust.csv").set_index("dYLUT")
_TDC_NORM_DF_ULTRA45 = pd.read_csv(
    BASE_PATH / "ultra45_tdc_norm.csv", header=1, index_col="Index"
)
_TDC_NORM_DF_ULTRA90 = pd.read_csv(
    BASE_PATH / "ultra90_tdc_norm.csv", header=1, index_col="Index"
)
_BACK_POS_DF_ULTRA45 = pd.read_csv(
    BASE_PATH / "ultra45_back-pos-luts.csv", index_col="Index_offset"
)
_BACK_POS_DF_ULTRA90 = pd.read_csv(
    BASE_PATH / "ultra90_back-pos-luts.csv", index_col="Index_offset"
)
_ENERGY_NORM_DF = pd.read_csv(BASE_PATH / "EgyNorm.mem.csv")
_IMAGE_PARAMS_DF = pd.read_csv(BASE_PATH / "FM45_Startup1_ULTRA_IMGPARAMS_20240719.csv")


def get_y_adjust(dy_lut: np.ndarray) -> npt.NDArray:
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
    return _YADJUST_DF["dYAdj"].iloc[dy_lut].values


def get_norm(dn: xr.DataArray, key: str, file_label: str) -> npt.NDArray:
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
    file_label : str
        Instrument (ultra45 or ultra90).

    Returns
    -------
    dn_norm : np.ndarray
        Normalized DNs.
    """
    if file_label == "ultra45":
        tdc_norm_df = _TDC_NORM_DF_ULTRA45
    else:
        tdc_norm_df = _TDC_NORM_DF_ULTRA90

    dn_norm = tdc_norm_df[key].iloc[dn].values

    return dn_norm


def get_back_position(back_index: np.ndarray, key: str, file_label: str) -> npt.NDArray:
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
    file_label : str
        Instrument (ultra45 or ultra90).

    Returns
    -------
    dn_converted : np.ndarray
        Converted DNs to Units of hundredths of a millimeter.
    """
    if file_label == "ultra45":
        back_pos_df = _BACK_POS_DF_ULTRA45
    else:
        back_pos_df = _BACK_POS_DF_ULTRA90

    return back_pos_df[key].values[back_index]


def get_energy_norm(ssd: np.ndarray, composite_energy: np.ndarray) -> npt.NDArray:
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
    row_number = ssd * 4096 + composite_energy

    return _ENERGY_NORM_DF["NormEnergy"].iloc[row_number]


def get_image_params(image: str) -> np.float64:
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
    value: np.float64 = _IMAGE_PARAMS_DF[image].values[0]
    return value

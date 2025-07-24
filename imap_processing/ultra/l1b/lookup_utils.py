"""Contains tools for lookup tables for l1b."""

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from imap_processing.quality_flags import ImapDEUltraFlags


def get_y_adjust(dy_lut: np.ndarray, ancillary_files: dict) -> npt.NDArray:
    """
    Adjust the front yf position based on the particle's trajectory.

    Instead of using trigonometry, this function utilizes a 256-element lookup table
    to find the Y adjustment. For more details, refer to pages 37-38 of the
    IMAP-Ultra Flight Software Specification document.

    Parameters
    ----------
    dy_lut : np.ndarray
        Change in y direction used for the lookup table (mm).
    ancillary_files : dict[Path]
        Ancillary files containing the lookup tables.

    Returns
    -------
    yadj : np.ndarray
        Y adjustment (mm).
    """
    yadjust_df = pd.read_csv(ancillary_files["l1b-yadjust-lookup"]).set_index("dYLUT")
    return yadjust_df["dYAdj"].iloc[dy_lut].values


def get_norm(
    dn: xr.DataArray, key: str, file_label: str, ancillary_files: dict
) -> npt.NDArray:
    """
    Correct mismatches between the stop Time to Digital Converters (TDCs).

    There are mismatches between the stop TDCs, i.e., SpN, SpS, SpE, and SpW.
    Before these can be used, they must be corrected, or normalized,
    using lookup tables.

    Further description is available on pages 31-32 of the IMAP-Ultra Flight Software
    Specification document. This will work for both Tp{key}Norm,
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
    ancillary_files : dict[Path]
        Ancillary files containing the lookup tables.

    Returns
    -------
    dn_norm : np.ndarray
        Normalized DNs.
    """
    if file_label == "ultra45":
        tdc_norm_df = pd.read_csv(
            ancillary_files["l1b-45sensor-tdc-norm-lookup"], header=1, index_col="Index"
        )
    else:
        tdc_norm_df = pd.read_csv(
            ancillary_files["l1b-90sensor-tdc-norm-lookup"], header=1, index_col="Index"
        )

    dn_norm = tdc_norm_df[key].iloc[dn].values

    return dn_norm


def get_back_position(
    back_index: np.ndarray, key: str, file_label: str, ancillary_files: dict
) -> npt.NDArray:
    """
    Convert normalized TDC values using lookup tables.

    The anodes behave non-linearly near their edges; thus, the use of lookup tables
    instead of linear equations is necessary. The computation will use different
    tables to accommodate variations between the top and bottom anodes.
    Further description is available on page 32 of the
    IMAP-Ultra Flight Software Specification document.

    Parameters
    ----------
    back_index : np.ndarray
        Options include SpSNorm - SpNNorm + 2047, SpENorm - SpWNorm + 2047,
        SpSNorm - SpNNorm + 2047, or SpENorm - SpWNorm + 2047.
    key : str
        XBkTp, YBkTp, XBkBt, or YBkBt.
    file_label : str
        Instrument (ultra45 or ultra90).
    ancillary_files : dict[Path]
        Ancillary files containing the lookup tables.

    Returns
    -------
    dn_converted : np.ndarray
        Converted DNs to Units of hundredths of a millimeter.
    """
    if file_label == "ultra45":
        back_pos_df = pd.read_csv(
            ancillary_files["l1b-45sensor-back-pos-lookup"], index_col="Index_offset"
        )
    else:
        back_pos_df = pd.read_csv(
            ancillary_files["l1b-90sensor-back-pos-lookup"], index_col="Index_offset"
        )

    return back_pos_df[key].values[back_index]


def get_energy_norm(
    ssd: np.ndarray, composite_energy: np.ndarray, ancillary_files: dict
) -> npt.NDArray:
    """
    Normalize composite energy per SSD using a lookup table.

    Further description is available on page 41 of the
    IMAP-Ultra Flight Software Specification document.
    Note : There are 8 SSDs containing
    4096 composite energies each.

    Parameters
    ----------
    ssd : np.ndarray
        Acts as index 1.
    composite_energy : np.ndarray
        Acts as index 2.
    ancillary_files : dict[Path]
        Ancillary files containing the lookup tables.

    Returns
    -------
    norm_composite_energy : np.ndarray
        Normalized composite energy.
    """
    row_number = ssd * 4096 + composite_energy
    norm_lookup = pd.read_csv(ancillary_files["l1b-egynorm-lookup"])
    return norm_lookup["NormEnergy"].iloc[row_number]


def get_image_params(image: str, sensor: str, ancillary_files: dict) -> np.float64:
    """
    Lookup table for image parameters.

    Further description is available starting on
    page 30 of the IMAP-Ultra Flight Software
    Specification document.

    Parameters
    ----------
    image : str
        The column name to lookup in the CSV file, e.g., 'XFTLTOFF' or 'XFTRTOFF'.
    sensor : str
        Sensor name: "ultra45" or "ultra90".
    ancillary_files : dict[Path]
        Ancillary files containing the lookup tables.

    Returns
    -------
    value : np.float64
        Image parameter value from the CSV file.
    """
    if sensor == "ultra45":
        lookup_table = pd.read_csv(ancillary_files["l1b-45sensor-imgparams-lookup"])
    else:
        lookup_table = pd.read_csv(ancillary_files["l1b-90sensor-imgparams-lookup"])

    value: np.float64 = lookup_table[image].values[0]
    return value


def get_angular_profiles(
    start_type: str, sensor: str, ancillary_files: dict
) -> pd.DataFrame:
    """
    Lookup table for FWHM for theta and phi.

    Further description is available starting on
    page 18 of the Algorithm Document.

    Parameters
    ----------
    start_type : str
       Start Type: Left, Right.
    sensor : str
        Sensor name: "ultra45" or "ultra90".
    ancillary_files : dict[Path]
        Ancillary files.

    Returns
    -------
    lookup_table : DataFrame
        Angular profile lookup table for a given start_type and sensor.
    """
    lut_descriptor = f"l1b-{sensor[-2:]}sensor-{start_type.lower()}slit-lookup"
    lookup_table = pd.read_csv(ancillary_files[lut_descriptor])

    return lookup_table


def get_energy_efficiencies(ancillary_files: dict) -> pd.DataFrame:
    """
    Lookup table for efficiencies for theta and phi.

    Further description is available starting on
    page 18 of the Algorithm Document.

    Parameters
    ----------
    ancillary_files : dict[Path]
        Ancillary files.

    Returns
    -------
    lookup_table : DataFrame
        Efficiencies lookup table for a given sensor.
    """
    # TODO: add sensor to input when new lookup tables are available.
    lookup_table = pd.read_csv(ancillary_files["l1b-45sensor-logistic-interpolation"])

    return lookup_table


def get_geometric_factor(
    ancillary_files: dict,
    filename: str,
    phi: NDArray,
    theta: NDArray,
    quality_flag: NDArray,
) -> tuple[NDArray, NDArray]:
    """
    Lookup table for geometric factor using nearest neighbor.

    Parameters
    ----------
    ancillary_files : dict[Path]
        Ancillary files.
    filename : str
        Name of the file in ancillary_files to use.
    phi : NDArray
        Azimuth angles in degrees.
    theta : NDArray
        Elevation angles in degrees.
    quality_flag : NDArray
        Quality flag to set when geometric factor is zero.

    Returns
    -------
    geometric_factor : NDArray
        Geometric factor.
    """
    gf_table = pd.read_csv(
        ancillary_files[filename], header=None, skiprows=6, nrows=301
    ).to_numpy(dtype=float)
    theta_table = pd.read_csv(
        ancillary_files[filename], header=None, skiprows=308, nrows=301
    ).to_numpy(dtype=float)
    phi_table = pd.read_csv(
        ancillary_files[filename], header=None, skiprows=610, nrows=301
    ).to_numpy(dtype=float)

    # Assume uniform grids: extract 1D arrays from first row/col
    theta_vals = theta_table[0, :]  # columns represent theta
    phi_vals = phi_table[:, 0]  # rows represent phi

    # Find nearest index in table for each input value
    phi_idx = np.abs(phi_vals[:, None] - phi).argmin(axis=0)
    theta_idx = np.abs(theta_vals[:, None] - theta).argmin(axis=0)

    # Fetch geometric factor values at nearest (phi, theta) pairs
    geometric_factor = gf_table[phi_idx, theta_idx]

    phi_rad = np.deg2rad(phi)
    numerator = 5.0 * np.cos(phi_rad)
    denominator = 1 + 2.80 * np.cos(phi_rad)
    # Equation 19 in the Ultra Algorithm Document.
    theta_nom = np.arctan(numerator / denominator)
    theta_nom = np.rad2deg(theta_nom)

    outside_fov = np.abs(theta) > theta_nom
    quality_flag[outside_fov] |= ImapDEUltraFlags.FOV.value

    return geometric_factor


def get_ph_corrected(
    sensor: str,
    location: str,
    ancillary_files: dict,
    xlut: NDArray,
    ylut: NDArray,
    quality_flag: NDArray,
) -> tuple[NDArray, NDArray]:
    """
    PH correction for stop anodes, top and bottom.

    Further description is available starting on
    page 207 of the Ultra Flight Software Document.

    Parameters
    ----------
    sensor : str
        Sensor name: "ultra45" or "ultra90".
    location : str
        Location: "tp" or "bt".
    ancillary_files : dict[Path]
        Ancillary files.
    xlut : NDArray
        X lookup index for PH correction.
    ylut : NDArray
        Y lookup index for PH correction.
    quality_flag : NDArray
        Quality flag to set when there is an outlier.

    Returns
    -------
    ph_correction : NDArray
        Correction for pulse height.
    quality_flag : NDArray
        Quality flag updated with PH correction flags.
    """
    ph_correct = pd.read_csv(
        ancillary_files[f"l1b-{sensor[-2:]}sensor-sp{location}phcorr"], header=None
    )
    ph_correct_array = ph_correct.to_numpy()

    max_x, max_y = ph_correct_array.shape[0] - 1, ph_correct_array.shape[1] - 1

    # Clamp indices to nearest valid value
    xlut_clamped = np.clip(xlut.astype(int), 0, max_x)
    ylut_clamped = np.clip(ylut.astype(int), 0, max_y)

    # Flag where clamping occurred
    flagged_mask = (xlut != xlut_clamped) | (ylut != ylut_clamped)
    quality_flag[flagged_mask] |= ImapDEUltraFlags.PHCORR.value

    ph_correction = ph_correct_array[xlut_clamped, ylut_clamped]

    return ph_correction, quality_flag

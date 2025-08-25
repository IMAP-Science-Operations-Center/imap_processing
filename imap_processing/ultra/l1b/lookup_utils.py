"""Contains tools for lookup tables for l1b."""

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from imap_processing.quality_flags import ImapDEOutliersUltraFlags
from imap_processing.ultra.constants import UltraConstants

logger = logging.getLogger(__name__)


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

    outside_fov = ~is_inside_fov(np.deg2rad(phi), np.deg2rad(theta))
    quality_flag[outside_fov] |= ImapDEOutliersUltraFlags.FOV.value

    return geometric_factor


def get_scattering_coefficients(
    ancillary_files: dict,
    instrument_id: int,
    theta: NDArray,
    phi: NDArray,
) -> tuple[NDArray, NDArray]:
    """
    Get a and g coefficients for theta and phi to compute scattering FWHM.

    Parameters
    ----------
    ancillary_files : dict[Path]
        Ancillary files.
    instrument_id : int
        Instrument ID, either 45 or 90.
    theta : NDArray
        Elevation angles in degrees.
    phi : NDArray
        Azimuth angles in degrees.

    Returns
    -------
    tuple
        Scattering a and g values corresponding to the given theta and phi values.
    """
    # TODO remove the line below when the 45 sensor scattering coefficients are
    #   delivered.
    instrument_id = 90
    descriptor = f"l1b-{instrument_id}sensor-scattering-calibration"
    theta_grid = pd.read_csv(
        ancillary_files[descriptor], header=None, skiprows=7, nrows=241
    ).to_numpy(dtype=float)
    phi_grid = pd.read_csv(
        ancillary_files[descriptor], header=None, skiprows=249, nrows=241
    ).to_numpy(dtype=float)
    a_theta = pd.read_csv(
        ancillary_files[descriptor], header=None, skiprows=491, nrows=241
    ).to_numpy(dtype=float)
    g_theta = pd.read_csv(
        ancillary_files[descriptor], header=None, skiprows=733, nrows=241
    ).to_numpy(dtype=float)
    a_phi = pd.read_csv(
        ancillary_files[descriptor], header=None, skiprows=975, nrows=241
    ).to_numpy(dtype=float)
    g_phi = pd.read_csv(
        ancillary_files[descriptor], header=None, skiprows=1217, nrows=241
    ).to_numpy(dtype=float)

    # Assume uniform grids: extract 1D arrays from first row/col
    theta_vals = theta_grid[0, :]  # columns represent theta
    phi_vals = phi_grid[:, 0]  # rows represent phi

    # Find nearest index in table for each input value
    phi_idx = np.abs(phi_vals[:, None] - phi).argmin(axis=0)
    theta_idx = np.abs(theta_vals[:, None] - theta).argmin(axis=0)

    # Fetch a and g values at nearest (phi, theta) pairs
    a_theta_val = a_theta[phi_idx, theta_idx]
    g_theta_val = g_theta[phi_idx, theta_idx]
    a_phi_val = a_phi[phi_idx, theta_idx]
    g_phi_val = g_phi[phi_idx, theta_idx]

    return np.column_stack([a_theta_val, g_theta_val]), np.column_stack(
        [a_phi_val, g_phi_val]
    )


def mask_below_fwhm_scattering_threshold(
    theta_coeffs: np.ndarray,
    phi_coeffs: np.ndarray,
    energy: int,
) -> np.ndarray:
    """
    Determine indices of theta and phi values below the FWHM scattering threshold.

    For each phi and theta, calculate the FWHM using the formula:
    FWHM = A*E^g
    If Phi FWHM or Theta FWHM > the scattering requirements from the table above,
    mask the instrument frame pixel.

    Parameters
    ----------
    theta_coeffs : NDArray
        Coefficients for theta FWHM calculation (a and g) for each pixel.
    phi_coeffs : NDArray
        Coefficients for phi FWHM calculation (a and g) for each pixel.
    energy : int
        Energy in keV.

    Returns
    -------
    numpy.ndarray
        Boolean array indicating incides below the scattering threshold.
    """
    scattering_thresholds = UltraConstants.ULTRA_FWHM_SCATTERING_CULLING_THRESHOLDS
    # Calculate FWHM for theta and phi
    fwhm_theta = theta_coeffs[..., 0] * energy ** theta_coeffs[..., 1]
    fwhm_phi = phi_coeffs[..., 0] * energy ** phi_coeffs[..., 1]

    try:
        # Get the scattering threshold based on the energy
        threshold = next(
            threshold
            for energy_range, threshold in scattering_thresholds.items()
            if energy_range[0] <= energy < energy_range[1]
        )
    except StopIteration:
        logger.warning(
            f"Energy {energy} keV is out of bounds for scattering thresholds. Using "
            f"zero for as threshold."
        )
        threshold = 0
    # Combine conditions for both theta and phi
    return np.logical_and(fwhm_theta <= threshold, fwhm_phi <= threshold)


def get_nominal_for_by_spin_phase(
    ancillary_files: dict, instrument_id: int
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Get indices of pixels in the nominal FOR as a function of spin phase.

    This function also returns the theta / phi values in the instrument frame and
    right ascension / declination values in the IMAP frame.

    Parameters
    ----------
    ancillary_files : dict[Path]
        Ancillary files.
    instrument_id : int
        Instrument ID, either 45 or 90.

    Returns
    -------
    tuple
        Scattering a and g values corresponding to the given theta and phi values.
    """
    # TODO replace with actual lookup table when available.
    descriptor = f"l1c-{instrument_id}sensor-nominal-for-lookup"
    filename = ancillary_files[descriptor]

    calibration_data = pd.read_csv(filename, header=None, skiprows=1).to_numpy(
        dtype=float
    )
    ra_and_dec = calibration_data[:, :2]  # Shape (npix, 2)
    theta_and_phi = np.random.randint(-60, 60, size=ra_and_dec.shape)  # Shape (npix, 2)
    # This array indicates whether each pixel is in the nominal FOR at each spin phase
    # step (15000 steps for a full rotation with 1 ms resolution).
    for_indices_by_spin_phase = calibration_data[:, 2:].astype(
        bool
    )  # Shape (npix, 15000)
    return for_indices_by_spin_phase, theta_and_phi, ra_and_dec


def is_inside_fov(phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Determine angles in the field of view (FOV).

    This function is used in the deadtime correction to determine whether a given
    (theta, phi) angle is within the instrument's Field of View (FOV).
    Only pixels inside the FOV are considered for time accumulation. The FOV boundary
    is defined by equation 19 in the Ultra Algorithm Document.

    Parameters
    ----------
    phi : np.ndarray
        Azimuth angles in radians.
    theta : np.ndarray
        Elevation angles in radians.

    Returns
    -------
    numpy.ndarray
        Boolean array indicating if the angle is in the FOV, False otherwise.
    """
    numerator = 5.0 * np.cos(phi)
    denominator = 1 + 2.80 * np.cos(phi)
    # Equation 19 in the Ultra Algorithm Document.
    theta_nom = np.arctan(numerator / denominator)
    return np.abs(theta) <= theta_nom


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
    quality_flag[flagged_mask] |= ImapDEOutliersUltraFlags.PHCORR.value

    ph_correction = ph_correct_array[xlut_clamped, ylut_clamped]

    return ph_correction, quality_flag


def get_ebins(
    lut: str,
    energy: NDArray,
    ctof: NDArray,
    ebins: NDArray,
    ancillary_files: dict,
) -> NDArray:
    """
    Get energy bins from the lookup table.

    Parameters
    ----------
    lut : str
        Lookup table name, e.g., "l1b-tofxpht".
    energy : NDArray
        Energy from the event (keV).
    ctof : NDArray
        Corrected TOF (tenths of a ns).
    ebins : NDArray
        Energy bins to fill with values.
    ancillary_files : dict[Path]
        Ancillary files.

    Returns
    -------
    ebins : NDArray
        Energy bins from the lookup table.
    """
    with open(ancillary_files[lut]) as f:
        all_lines = f.readlines()
        pixel_text = "".join(all_lines[4:])

    lut_array = np.fromstring(pixel_text, sep=" ", dtype=int).reshape((2048, 4096))
    # Note that the LUT is indexed [energy, ctof] for l1b-tofxph
    # and [ctof, energy] for everything else.
    if lut == "l1b-tofxph":
        energy_lookup = (2048 - np.floor(energy)).astype(int)
        ctof_lookup = np.floor(ctof).astype(int)
        valid = (
            (energy_lookup >= 0)
            & (energy_lookup < 2048)
            & (ctof_lookup >= 0)
            & (ctof_lookup < 4096)
        )
        ebins[valid] = lut_array[energy_lookup[valid], ctof_lookup[valid]]
    else:
        energy_lookup = np.floor(energy).astype(int)
        ctof_lookup = (2048 - np.floor(ctof)).astype(int)
        valid = (
            (energy_lookup >= 0)
            & (energy_lookup < 4096)
            & (ctof_lookup >= 0)
            & (ctof_lookup < 2048)
        )
        ebins[valid] = lut_array[ctof_lookup[valid], energy_lookup[valid]]

    return ebins

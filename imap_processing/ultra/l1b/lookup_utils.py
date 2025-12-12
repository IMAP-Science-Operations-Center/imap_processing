"""Contains tools for lookup tables for l1b."""

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from imap_processing.quality_flags import ImapDEOutliersUltraFlags
from imap_processing.ultra.constants import UltraConstants


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


def load_geometric_factor_tables(
    ancillary_files: dict,
    filename: str,
) -> dict:
    """
    Lookup tables for geometric factor.

    Parameters
    ----------
    ancillary_files : dict[Path]
        Ancillary files.
    filename : str
        Name of the file in ancillary_files to use.

    Returns
    -------
    geometric_factor_tables : dict
        Geometric factor lookup tables.
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

    return {
        "gf_table": gf_table,
        "theta_table": theta_table,
        "phi_table": phi_table,
    }


def get_geometric_factor(
    phi: NDArray,
    theta: NDArray,
    quality_flag: NDArray,
    ancillary_files: dict | None = None,
    filename: str | None = None,
    geometric_factor_tables: dict | None = None,
) -> tuple[NDArray, NDArray]:
    """
    Lookup table for geometric factor using nearest neighbor.

    Parameters
    ----------
    phi : NDArray
        Azimuth angles in degrees.
    theta : NDArray
        Elevation angles in degrees.
    quality_flag : NDArray
        Quality flag to set when geometric factor is zero.
    ancillary_files : dict[Path], optional
        Ancillary files.
    filename : str, optional
        Name of the file in ancillary_files to use.
    geometric_factor_tables : dict, optional
        Preloaded geometric factor lookup tables. If not provided, will load.

    Returns
    -------
    geometric_factor : NDArray
        Geometric factor.
    """
    if geometric_factor_tables is None:
        if ancillary_files is None or filename is None:
            raise ValueError(
                "ancillary_files and filename must be provided if "
                "geometric_factor_tables is not supplied."
            )
        geometric_factor_tables = load_geometric_factor_tables(
            ancillary_files, filename
        )
    # Assume uniform grids: extract 1D arrays from first row/col
    theta_vals = geometric_factor_tables["theta_table"][0, :]  # columns represent theta
    phi_vals = geometric_factor_tables["phi_table"][:, 0]  # rows represent phi

    # Find nearest index in table for each input value
    phi_idx = np.abs(phi_vals[:, None] - phi).argmin(axis=0)
    theta_idx = np.abs(theta_vals[:, None] - theta).argmin(axis=0)

    # Fetch geometric factor values at nearest (phi, theta) pairs
    geometric_factor = geometric_factor_tables["gf_table"][phi_idx, theta_idx]

    outside_fov = ~is_inside_fov(np.deg2rad(theta), np.deg2rad(phi))
    quality_flag[outside_fov] |= ImapDEOutliersUltraFlags.FOV.value

    return geometric_factor


def load_scattering_lookup_tables(ancillary_files: dict, instrument_id: int) -> dict:
    """
    Load scattering coefficient lookup tables for the specified instrument.

    Parameters
    ----------
    ancillary_files : dict
        Ancillary files.
    instrument_id : int
        Instrument ID, either 45 or 90.

    Returns
    -------
    dict
        Dictionary containing arrays for theta_grid, phi_grid, a_theta, g_theta,
         a_phi, g_phi.
    """
    # TODO remove the line below when the 45 sensor scattering coefficients are
    #  delivered.
    instrument_id = 90
    descriptor = f"l1b-{instrument_id}sensor-scattering-calibration-data"
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
    return {
        "theta_grid": theta_grid,
        "phi_grid": phi_grid,
        "a_theta": a_theta,
        "g_theta": g_theta,
        "a_phi": a_phi,
        "g_phi": g_phi,
    }


def get_scattering_coefficients(
    theta: NDArray,
    phi: NDArray,
    lookup_tables: dict | None = None,
    ancillary_files: dict | None = None,
    instrument_id: int | None = None,
) -> tuple[NDArray, NDArray]:
    """
    Get a and g coefficients for theta and phi to compute scattering FWHM.

    Parameters
    ----------
    theta : NDArray
        Elevation angles in degrees.
    phi : NDArray
        Azimuth angles in degrees.
    lookup_tables : dict, optional
        Preloaded lookup tables. If not provided, will load using ancillary_files and
         instrument_id.
    ancillary_files : dict, optional
        Ancillary files, required if lookup_tables is not provided.
    instrument_id : int, optional
        Instrument ID, required if lookup_tables is not provided.

    Returns
    -------
    tuple
        Scattering a and g values corresponding to the given theta and phi values.
    """
    if lookup_tables is None:
        if ancillary_files is None or instrument_id is None:
            raise ValueError(
                "ancillary_files and instrument_id must be provided if lookup_tables "
                "is not supplied."
            )
        lookup_tables = load_scattering_lookup_tables(ancillary_files, instrument_id)

    theta_grid = lookup_tables["theta_grid"]
    phi_grid = lookup_tables["phi_grid"]
    a_theta = lookup_tables["a_theta"]
    g_theta = lookup_tables["g_theta"]
    a_phi = lookup_tables["a_phi"]
    g_phi = lookup_tables["g_phi"]

    theta_vals = theta_grid[0, :]  # columns represent theta
    phi_vals = phi_grid[:, 0]  # rows represent phi

    phi_idx = np.abs(phi_vals[:, None] - phi).argmin(axis=0)
    theta_idx = np.abs(theta_vals[:, None] - theta).argmin(axis=0)

    a_theta_val = a_theta[phi_idx, theta_idx]
    g_theta_val = g_theta[phi_idx, theta_idx]
    a_phi_val = a_phi[phi_idx, theta_idx]
    g_phi_val = g_phi[phi_idx, theta_idx]

    return np.column_stack([a_theta_val, g_theta_val]), np.column_stack(
        [a_phi_val, g_phi_val]
    )


def is_inside_fov(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Determine angles in the field of view (FOV).

    This function is used in the deadtime correction to determine whether a given
    (theta, phi) angle is within the instrument's Field of View (FOV).
    Only pixels inside the FOV are considered for time accumulation. The FOV boundary
    is defined by equation 19 in the Ultra Algorithm Document.

    Parameters
    ----------
    theta : np.ndarray
        Elevation angles in radians.
    phi : np.ndarray
        Azimuth angles in radians.

    Returns
    -------
    numpy.ndarray
        Boolean array indicating if the angle is in the FOV, False otherwise.
    """
    numerator = 5.0 * np.cos(phi)
    denominator = 1.0 + 2.80 * np.cos(phi)
    # Equation 19 in the Ultra Algorithm Document.
    theta_nom = np.arctan(numerator / denominator) - np.radians(
        UltraConstants.FOV_THETA_OFFSET_DEG
    )

    theta_check = np.abs(theta) <= np.abs(theta_nom)
    phi_check = np.abs(phi) <= np.radians(UltraConstants.FOV_PHI_LIMIT_DEG)

    return theta_check & phi_check


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


def get_scattering_thresholds(ancillary_files: dict) -> dict:
    """
    Load scattering culling thresholds as a function of energy from a lookup table.

    Parameters
    ----------
    ancillary_files : dict[Path]
        Ancillary files.

    Returns
    -------
    threshold_dict
         Dictionary containing energy ranges and the corresponding scattering culling
          threshold.
    """
    # Culling FWHM Scattering values as a function of energy.
    thresholds = pd.read_csv(
        ancillary_files["l1b-scattering-thresholds-per-energy"], header=None, skiprows=1
    ).to_numpy(dtype=np.float64)
    # The first two columns represent the energy range (min, max) in keV, and the
    # value is the FWHM scattering threshold in degrees
    threshold_dict = {(row[0], row[1]): row[2] for row in thresholds}

    return threshold_dict

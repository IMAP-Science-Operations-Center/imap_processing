"""Calculates Extended Raw Events for ULTRA L1b."""

# TODO: Come back and add in FSW logic.
import logging
from collections import namedtuple
from enum import Enum

import numpy as np
import pandas
import xarray
from numpy import ndarray
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

from imap_processing.quality_flags import ImapDEOutliersUltraFlags
from imap_processing.spice.spin import get_spin_data
from imap_processing.spice.time import sct_to_et
from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1b.lookup_utils import (
    get_angular_profiles,
    get_back_position,
    get_ebins,
    get_energy_efficiencies,
    get_energy_norm,
    get_image_params,
    get_norm,
    get_ph_corrected,
    get_y_adjust,
)

logger = logging.getLogger(__name__)

FILLVAL_UINT8 = 255
FILLVAL_FLOAT32 = -1.0e31
FILLVAL_FLOAT64 = -1.0e31


class StartType(Enum):
    """Start Type: 1=Left, 2=Right."""

    Left = 1
    Right = 2


class StopType(Enum):
    """Stop Type: 1=Top, 2=Bottom, SSD: 8-15."""

    Top = 1
    Bottom = 2
    PH = [1, 2]  # noqa RUF012 mutable class attribute
    SSD = [8, 9, 10, 11, 12, 13, 14, 15]  # noqa RUF012 mutable class attribute


class CoinType(Enum):
    """Coin Type: 1=Top, 2=Bottom."""

    Top = 1
    Bottom = 2


PHTOFResult = namedtuple("PHTOFResult", ["tof", "t2", "xb", "yb", "tofx", "tofy"])


def get_front_x_position(
    start_type: ndarray, start_position_tdc: ndarray, sensor: str, ancillary_files: dict
) -> ndarray:
    """
    Calculate the front xf position.

    Converts Start Position Time to Digital Converter (TDC)
    values into units of hundredths of a millimeter using a scale factor and offsets.
    Further description is available on pages 30 of
    IMAP-Ultra Flight Software Specification document.

    Parameters
    ----------
    start_type : ndarray
        Start Type: 1=Left, 2=Right.
    start_position_tdc : ndarray
        Start Position Time to Digital Converter (TDC).
    sensor : str
        Sensor name.
    ancillary_files : dict[Path]
        Ancillary files containing the lookup tables.

    Returns
    -------
    xf : ndarray
        X front position (hundredths of a millimeter).
    """
    # Left and right start types.
    indices = np.nonzero((start_type == 1) | (start_type == 2))

    xftsc = get_image_params("XFTSC", sensor, ancillary_files)
    xft_lt_off = get_image_params("XFTLTOFF", sensor, ancillary_files)
    xft_rt_off = get_image_params("XFTRTOFF", sensor, ancillary_files)
    xft_off = np.where(start_type[indices] == 1, xft_lt_off, xft_rt_off)

    # Calculate xf and convert to hundredths of a millimeter
    xf: ndarray = (xftsc * -start_position_tdc[indices] + xft_off) * 100

    return xf


def get_front_y_position(
    start_type: ndarray, yb: ndarray, ancillary_files: dict
) -> tuple[ndarray, ndarray]:
    """
    Compute the adjustments for the front y position and distance front to back.

    This function utilizes lookup tables and trigonometry based on
    the angle of the foil. Further description is available in the
    IMAP-Ultra Flight Software Specification document pg 30.

    Parameters
    ----------
    start_type : np.array
        Start Type: 1=Left, 2=Right.
    yb : np.array
        Y back position in hundredths of a millimeter.
    ancillary_files : dict[Path]
        Ancillary files containing the lookup tables.

    Returns
    -------
    d : np.array
        Distance front to back in hundredths of a millimeter.
    yf : np.array
        Front y position in hundredths of a millimeter.
    """
    # Determine start types
    index_left = np.nonzero(start_type == 1)
    index_right = np.nonzero(start_type == 2)

    yf = np.zeros(len(start_type))
    d = np.zeros(len(start_type))

    # Compute adjustments for left start type
    dy_lut_left = np.floor(
        (UltraConstants.YF_ESTIMATE_LEFT - yb[index_left] / 100)
        * UltraConstants.N_ELEMENTS
        / UltraConstants.TRIG_CONSTANT
        + 0.5
    )
    # y adjustment in mm
    y_adjust_left = get_y_adjust(dy_lut_left, ancillary_files) / 100
    # hundredths of a millimeter
    yf[index_left] = (UltraConstants.YF_ESTIMATE_LEFT - y_adjust_left) * 100
    # distance adjustment in mm
    distance_adjust_left = np.sqrt(2) * UltraConstants.D_SLIT_FOIL - y_adjust_left
    # hundredths of a millimeter
    d[index_left] = (UltraConstants.SLIT_Z - distance_adjust_left) * 100

    # Compute adjustments for right start type
    dy_lut_right = np.floor(
        (yb[index_right] / 100 - UltraConstants.YF_ESTIMATE_RIGHT)
        * UltraConstants.N_ELEMENTS
        / UltraConstants.TRIG_CONSTANT
        + 0.5
    )
    # y adjustment in mm
    y_adjust_right = get_y_adjust(dy_lut_right, ancillary_files) / 100
    # hundredths of a millimeter
    yf[index_right] = (UltraConstants.YF_ESTIMATE_RIGHT + y_adjust_right) * 100
    # distance adjustment in mm
    distance_adjust_right = np.sqrt(2) * UltraConstants.D_SLIT_FOIL - y_adjust_right
    # hundredths of a millimeter
    d[index_right] = (UltraConstants.SLIT_Z - distance_adjust_right) * 100

    return np.array(d), np.array(yf)


def get_ph_tof_and_back_positions(
    de_dataset: xarray.Dataset, xf: np.ndarray, sensor: str, ancillary_files: dict
) -> PHTOFResult:
    """
    Calculate back xb, yb position and tof.

    An incoming particle may trigger pulses from one of the stop anodes.
    If so, four pulses are produced, one each from the north, south,
    east, and west sides.

    The Time Of Flight (tof) and the position of the particle at the
    back of the sensor are measured using the timing of the pulses.
    Further description is available on pages 32-33 of
    IMAP-Ultra Flight Software Specification document.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        Data in xarray format.
    xf : np.array
        X front position in (hundredths of a millimeter).
        Has same length as de_dataset.
    sensor : str
        Sensor name.
    ancillary_files : dict[Path]
        Ancillary files containing the lookup tables.

    Returns
    -------
    tof : np.array
        Time of flight (nanoseconds).
    t2 : np.array
        Particle time of flight from start to stop (tenths of a nanosecond).
    xb : np.array
        Back positions in x direction (hundredths of a millimeter).
    yb : np.array
        Back positions in y direction (hundredths of a millimeter).
    tofx : np.array
        X front position tof offset (tenths of a nanosecond).
    tofy : np.array
        Y front position tof offset (tenths of a nanosecond).
    """
    indices = np.nonzero(
        np.isin(de_dataset["stop_type"], [StopType.Top.value, StopType.Bottom.value])
    )[0]
    de_filtered = de_dataset.isel(epoch=indices)

    xf_ph = xf[indices]

    # There are mismatches between the stop TDCs, i.e., SpN, SpS, SpE, and SpW.
    # This normalizes the TDCs
    sp_n_norm = get_norm(
        de_filtered["stop_north_tdc"].data, "SpN", sensor, ancillary_files
    )
    sp_s_norm = get_norm(
        de_filtered["stop_south_tdc"].data, "SpS", sensor, ancillary_files
    )
    sp_e_norm = get_norm(
        de_filtered["stop_east_tdc"].data, "SpE", sensor, ancillary_files
    )
    sp_w_norm = get_norm(
        de_filtered["stop_west_tdc"].data, "SpW", sensor, ancillary_files
    )

    # Convert normalized TDC values into units of hundredths of a
    # millimeter using lookup tables.
    xb_index = sp_s_norm - sp_n_norm + 2047
    yb_index = sp_e_norm - sp_w_norm + 2047

    # Convert xf to a tof offset
    tofx = sp_n_norm + sp_s_norm
    tofy = sp_e_norm + sp_w_norm

    # tof is the average of the two tofs measured in the X and Y directions,
    # tofx and tofy
    # Units in tenths of a nanosecond
    t1 = tofx + tofy  # /2 incorporated into scale

    xb = np.zeros(len(indices))
    yb = np.zeros(len(indices))

    # particle_tof (t2) used later to compute etof
    t2 = np.zeros(len(indices))
    tof = np.zeros(len(indices))

    # Stop Type: 1=Top, 2=Bottom
    # Convert converts normalized TDC values into units of
    # hundredths of a millimeter using lookup tables.
    stop_type_top = de_filtered["stop_type"].data == StopType.Top.value
    xb[stop_type_top] = get_back_position(
        xb_index[stop_type_top], "XBkTp", sensor, ancillary_files
    )
    yb[stop_type_top] = get_back_position(
        yb_index[stop_type_top], "YBkTp", sensor, ancillary_files
    )

    # Correction for the propagation delay of the start anode and other effects.
    t2[stop_type_top] = get_image_params("TOFSC", sensor, ancillary_files) * t1[
        stop_type_top
    ] + get_image_params("TOFTPOFF", sensor, ancillary_files)
    # Variable xf_ph divided by 10 to convert to mm.
    tof[stop_type_top] = t2[stop_type_top] + xf_ph[
        stop_type_top
    ] / 10 * get_image_params("XFTTOF", sensor, ancillary_files)

    stop_type_bottom = de_filtered["stop_type"].data == StopType.Bottom.value
    xb[stop_type_bottom] = get_back_position(
        xb_index[stop_type_bottom], "XBkBt", sensor, ancillary_files
    )
    yb[stop_type_bottom] = get_back_position(
        yb_index[stop_type_bottom], "YBkBt", sensor, ancillary_files
    )

    # Correction for the propagation delay of the start anode and other effects.
    t2[stop_type_bottom] = get_image_params("TOFSC", sensor, ancillary_files) * t1[
        stop_type_bottom
    ] + get_image_params("TOFBTOFF", sensor, ancillary_files)  # 10*ns

    # Variable xf_ph divided by 10 to convert to mm.
    tof[stop_type_bottom] = t2[stop_type_bottom] + xf_ph[
        stop_type_bottom
    ] / 10 * get_image_params("XFTTOF", sensor, ancillary_files)

    return PHTOFResult(tof=tof, t2=t2, xb=xb, yb=yb, tofx=tofx, tofy=tofy)


def get_path_length(
    front_position: tuple, back_position: tuple, d: np.ndarray
) -> NDArray:
    """
    Calculate the path length.

    Parameters
    ----------
    front_position : tuple of floats
        Front position (xf,yf) (hundredths of a millimeter).
    back_position : tuple of floats
        Back position (xb,yb) (hundredths of a millimeter).
    d : np.ndarray
        Distance from slit to foil (hundredths of a millimeter).

    Returns
    -------
    path_length : np.ndarray
        Path length (r) (hundredths of a millimeter).
    """
    path_length = np.sqrt(
        (front_position[0] - back_position[0]) ** 2
        + (front_position[1] - back_position[1]) ** 2
        + (d) ** 2
    )

    return path_length


def get_ssd_back_position_and_tof_offset(
    de_dataset: xarray.Dataset, sensor: str, ancillary_files: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lookup the Y SSD positions (yb), TOF Offset, and SSD number.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        The input dataset containing STOP_TYPE and SSD_FLAG data.
    sensor : str
        Sensor name.
    ancillary_files : dict[Path]
        Ancillary files containing the lookup tables.

    Returns
    -------
    yb : np.ndarray
        Y SSD positions in hundredths of a millimeter.
    tof_offset : np.ndarray
        TOF offset.
    ssd_number : np.ndarray
        SSD number.

    Notes
    -----
    The X back position (xb) is assumed to be 0 for SSD.
    """
    indices = np.nonzero(np.isin(de_dataset["stop_type"], StopType.SSD.value))[0]
    de_filtered = de_dataset.isel(epoch=indices)

    yb = np.zeros(len(indices), dtype=np.float64)
    ssd_number = np.zeros(len(indices), dtype=int)
    tof_offset = np.zeros(len(indices), dtype=np.float64)

    for i in range(8):
        ssd_flag_mask = de_filtered[f"ssd_flag_{i}"].data == 1

        # Multiply ybs times 100 to convert to hundredths of a millimeter.
        yb[ssd_flag_mask] = (
            get_image_params(f"YBKSSD{i}", sensor, ancillary_files) * 100
        )
        ssd_number[ssd_flag_mask] = i

        tof_offset[
            (de_filtered["start_type"] == StartType.Left.value) & ssd_flag_mask
        ] = get_image_params(f"TOFSSDLTOFF{i}", sensor, ancillary_files)
        tof_offset[
            (de_filtered["start_type"] == StartType.Right.value) & ssd_flag_mask
        ] = get_image_params(f"TOFSSDRTOFF{i}", sensor, ancillary_files)

    return yb, tof_offset, ssd_number


def calculate_etof_xc(
    de_subset: xarray.Dataset,
    particle_tof: np.ndarray,
    sensor: str,
    location: str,
    ancillary_files: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the etof and xc values for the given subset.

    Parameters
    ----------
    de_subset : xarray.Dataset
        Subset of the dataset for a specific COIN_TYPE.
    particle_tof : np.ndarray
        Particle time of flight (i.e. from start to stop).
    sensor : str
        Sensor name.
    location : str
        Location indicator, either 'TP' (Top) or 'BT' (Bottom).
    ancillary_files : dict[Path]
        Ancillary files containing the lookup tables.

    Returns
    -------
    etof : np.ndarray
        Time for the electrons to travel back to the coincidence
        anode (tenths of a nanosecond).
    xc : np.ndarray
        X coincidence position (millimeters).
    """
    # CoinNNorm
    coin_n_norm = get_norm(
        de_subset["coin_north_tdc"], "CoinN", sensor, ancillary_files
    )
    # CoinSNorm
    coin_s_norm = get_norm(
        de_subset["coin_south_tdc"], "CoinS", sensor, ancillary_files
    )
    xc = get_image_params(f"XCOIN{location}SC", sensor, ancillary_files) * (
        coin_s_norm - coin_n_norm
    ) + get_image_params(f"XCOIN{location}OFF", sensor, ancillary_files)  # millimeter

    # Time for the electrons to travel back to coincidence anode.
    t2 = get_image_params("ETOFSC", sensor, ancillary_files) * (
        coin_n_norm + coin_s_norm
    ) + get_image_params(f"ETOF{location}OFF", sensor, ancillary_files)

    # Multiply by 10 to convert to tenths of a nanosecond.
    etof = t2 * 10 - particle_tof

    return etof, xc


def get_coincidence_positions(
    de_dataset: xarray.Dataset,
    particle_tof: np.ndarray,
    sensor: str,
    ancillary_files: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate coincidence positions.

    Calculate time for electrons to travel back to
    the coincidence anode (etof) and the x coincidence position (xc).

    The tof measured by the coincidence anode consists of the particle
    tof from start to stop, plus the time for the electrons to travel
    back to the coincidence anode.

    Further description is available on pages 34-35 of
    IMAP-Ultra Flight Software Specification document.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        Data in xarray format.
    particle_tof : np.ndarray
        Particle time of flight (i.e. from start to stop)
        (tenths of a nanosecond).
    sensor : str
        Sensor name.
    ancillary_files : dict[Path]
        Ancillary files containing the lookup tables.

    Returns
    -------
    etof : np.ndarray
        Time for the electrons to travel back to
        coincidence anode (tenths of a nanosecond).
    xc : np.ndarray
        X coincidence position (hundredths of a millimeter).
    """
    index_top = np.nonzero(np.isin(de_dataset["coin_type"], CoinType.Top.value))[0]
    de_top = de_dataset.isel(epoch=index_top)

    index_bottom = np.nonzero(np.isin(de_dataset["coin_type"], CoinType.Bottom.value))[
        0
    ]
    de_bottom = de_dataset.isel(epoch=index_bottom)

    etof = np.zeros(len(de_dataset["coin_type"]), dtype=np.float64)
    xc_array = np.zeros(len(de_dataset["coin_type"]), dtype=np.float64)

    # Normalized TDCs
    # For the stop anode, there are mismatches between the coincidence TDCs,
    # i.e., CoinN and CoinS. They must be normalized via lookup tables.
    etof_top, xc_top = calculate_etof_xc(
        de_top, particle_tof[index_top], sensor, "TP", ancillary_files
    )
    etof[index_top] = etof_top
    xc_array[index_top] = xc_top

    etof_bottom, xc_bottom = calculate_etof_xc(
        de_bottom, particle_tof[index_bottom], sensor, "BT", ancillary_files
    )
    etof[index_bottom] = etof_bottom
    xc_array[index_bottom] = xc_bottom

    # Convert to hundredths of a millimeter by multiplying times 100
    return etof, xc_array * 100


def get_de_velocity(
    front_position: tuple[NDArray, NDArray],
    back_position: tuple[NDArray, NDArray],
    d: np.ndarray,
    tof: np.ndarray,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Determine the direct event velocity.

    Parameters
    ----------
    front_position : tuple
        Front position (xf,yf) (hundredths of a millimeter).
    back_position : tuple
        Back position (xb,yb) (hundredths of a millimeter).
    d : np.array
        Distance from slit to foil (hundredths of a millimeter).
    tof : np.array
        Time of flight (tenths of a nanosecond).

    Returns
    -------
    velocities : np.ndarray
        N x 3 array of velocity components (vx, vy, vz) in km/s.
    v_hat : np.ndarray
        Unit vector in the direction of the velocity.
    r_hat : np.ndarray
        Position vector.
    """
    if tof[tof < 0].any():
        logger.info("Negative tof values found.")

    # distances in .1 mm
    delta_v = np.empty((len(d), 3), dtype=np.float32)
    delta_v[:, 0] = (front_position[0] - back_position[0]) * 0.1
    delta_v[:, 1] = (front_position[1] - back_position[1]) * 0.1
    delta_v[:, 2] = d * 0.1

    # Convert from 0.1mm/0.1ns to km/s.
    v_x = delta_v[:, 0] / tof * 1e3
    v_y = delta_v[:, 1] / tof * 1e3
    v_z = delta_v[:, 2] / tof * 1e3

    v_x[tof < 0] = FILLVAL_FLOAT32  # used as fillvals
    v_y[tof < 0] = FILLVAL_FLOAT32
    v_z[tof < 0] = FILLVAL_FLOAT32

    velocities = np.vstack((v_x, v_y, v_z)).T

    v_hat = velocities / np.linalg.norm(velocities, axis=1)[:, None]
    r_hat = -v_hat

    return velocities, -v_hat, -r_hat


def get_ssd_tof(
    de_dataset: xarray.Dataset, xf: np.ndarray, sensor: str, ancillary_files: dict
) -> NDArray[np.float64]:
    """
    Calculate back xb, yb position for the SSDs.

    An incoming particle could miss the stop anodes and instead
    hit one of the SSDs between the anodes. Which SSD is hit
    gives a coarse measurement of the y back position;
    the x back position will be fixed.

    Before hitting the SSD, particles pass through the stop foil;
    dislodged electrons are accelerated back towards the coincidence anode.
    The Coincidence Discrete provides a measure of the TOF.
    A scale factor and offsets, and a multiplier convert xf to a tof offset.

    Further description is available on pages 36 of
    IMAP-Ultra Flight Software Specification document.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        Data in xarray format.
    xf : np.array
        Front x position (hundredths of a millimeter).
    sensor : str
        Sensor name.
    ancillary_files : dict[Path]
        Ancillary files containing the lookup tables.

    Returns
    -------
    tof : np.ndarray
        Time of flight (tenths of a nanosecond).
    """
    _, tof_offset, ssd_number = get_ssd_back_position_and_tof_offset(
        de_dataset, sensor, ancillary_files
    )
    indices = np.nonzero(np.isin(de_dataset["stop_type"], [StopType.SSD.value]))[0]

    de_discrete = de_dataset.isel(epoch=indices)["coin_discrete_tdc"]

    time = (
        get_image_params("TOFSSDSC", sensor, ancillary_files) * de_discrete.values
        + tof_offset
    )

    # The scale factor and offsets, and a multiplier to convert xf to a tof offset.
    # Convert xf to mm by dividing by 100.
    tof = (
        time
        + get_image_params("TOFSSDTOTOFF", sensor, ancillary_files)
        + xf[indices] / 100 * get_image_params("XFTTOF", sensor, ancillary_files)
    ) * 10

    # Convert TOF to tenths of a nanosecond.
    return np.asarray(tof, dtype=np.float64)


def get_de_energy_kev(v: np.ndarray, species: np.ndarray) -> NDArray:
    """
    Calculate the direct event energy.

    Parameters
    ----------
    v : np.ndarray
        N x 3 array of velocity components (vx, vy, vz) in km/s.
    species : np.ndarray
        Species of the particle.

    Returns
    -------
    energy : np.ndarray
        Energy of the direct event in keV.
    """
    vv = v * 1e3  # convert km/s to m/s
    # Compute the sum of squares.
    v2 = np.sum(vv**2, axis=1)

    # Only compute where species == 1 and v is valid
    index_hydrogen = species == 1
    valid_velocity = np.isfinite(v2)
    valid_mask = index_hydrogen & valid_velocity

    energy = np.full_like(v2, FILLVAL_FLOAT32)

    # TODO: we will calculate the energies of the different species here.
    # 1/2 mv^2 in Joules, convert to keV
    energy[valid_mask] = (
        0.5 * UltraConstants.MASS_H * v2[valid_mask] * UltraConstants.J_KEV
    )

    return energy


def get_energy_pulse_height(
    stop_type: np.ndarray,
    energy: np.ndarray,
    xb: np.ndarray,
    yb: np.ndarray,
    sensor: str,
    ancillary_files: dict,
    quality_flags: NDArray,
) -> tuple[NDArray, NDArray]:
    """
    Calculate the pulse-height energy.

    Calculate energy measured using the
    pulse height from the stop anode.
    Lookup tables (lut) are used for corrections.
    Further description is available on pages 40-41 of
    IMAP-Ultra Flight Software Specification document.

    Parameters
    ----------
    stop_type : np.ndarray
        Stop type: 1=Top, 2=Bottom.
    energy : np.ndarray
        Energy measured using the pulse height.
    xb : np.ndarray
        X back position (hundredths of a millimeter).
    yb : np.ndarray
        Y back position (hundredths of a millimeter).
    sensor : str
        Sensor name.
    ancillary_files : dict[Path]
        Ancillary files containing the lookup tables.
    quality_flags : NDArray
        Quality flag to set when there is an outlier.

    Returns
    -------
    energy_ph : np.ndarray
        Energy measured using the pulse height
        from the stop anode (DN).
    """
    indices_top = np.where(stop_type == 1)[0]
    indices_bottom = np.where(stop_type == 2)[0]

    xlut = np.zeros(len(stop_type), dtype=np.float64)
    ylut = np.zeros(len(stop_type), dtype=np.float64)
    energy_ph = np.zeros(len(stop_type), dtype=np.float64)

    # Full-length correction arrays
    ph_correction = np.zeros(len(stop_type), dtype=np.float64)

    # Stop type 1
    xlut[indices_top] = (xb[indices_top] / 100 - 24.5 / 2) * 20 / 50  # mm
    ylut[indices_top] = (yb[indices_top] / 100 + 82 / 2) * 32 / 82  # mm
    # Stop type 2
    xlut[indices_bottom] = (xb[indices_bottom] / 100 + 50 + 24.5 / 2) * 20 / 50  # mm
    ylut[indices_bottom] = (yb[indices_bottom] / 100 + 82 / 2) * 32 / 82  # mm

    ph_correction_top, updated_flags_top = get_ph_corrected(
        "ultra45",
        "tp",
        ancillary_files,
        np.round(xlut[indices_top]),
        np.round(ylut[indices_top]),
        quality_flags[indices_top].copy(),
    )
    quality_flags[indices_top] = updated_flags_top
    ph_correction_bottom, updated_flags_bottom = get_ph_corrected(
        "ultra45",
        "bt",
        ancillary_files,
        np.round(xlut[indices_bottom]),
        np.round(ylut[indices_bottom]),
        quality_flags[indices_bottom].copy(),
    )
    quality_flags[indices_bottom] = updated_flags_bottom

    ph_correction[indices_top] = ph_correction_top / 1024
    ph_correction[indices_bottom] = ph_correction_bottom / 1024

    energy_ph[indices_top] = (
        (energy[indices_top] - get_image_params("SPTPPHOFF", sensor, ancillary_files))
        * ph_correction_top
        / 1024
    )

    energy_ph[indices_bottom] = (
        (
            energy[indices_bottom]
            - get_image_params("SPBTPHOFF", sensor, ancillary_files)
        )
        * ph_correction_bottom
        / 1024.0
    )

    return energy_ph, ph_correction


def get_energy_ssd(
    de_dataset: xarray.Dataset, ssd: np.ndarray, ancillary_files: dict
) -> NDArray[np.float64]:
    """
    Get SSD energy.

    For SSD events, the SSD itself provides a direct
    measurement of the energy. To cover higher energies,
    a so-called composite energy is calculated using the
    SSD energy and SSD energy pulse width.
    The result is then normalized per SSD via a lookup table.
    Further description is available on pages 41 of
    IMAP-Ultra Flight Software Specification document.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        Events dataset.
    ssd : np.ndarray
        SSD number.
    ancillary_files : dict[Path]
        Ancillary files containing the lookup tables.

    Returns
    -------
    energy_norm : np.ndarray
        Energy measured using the SSD.
    """
    ssd_indices = np.nonzero(np.isin(de_dataset["stop_type"], StopType.SSD.value))[0]
    energy = de_dataset["energy_ph"].data[ssd_indices]

    composite_energy = np.empty(len(energy), dtype=np.float64)

    composite_energy[energy >= UltraConstants.COMPOSITE_ENERGY_THRESHOLD] = (
        UltraConstants.COMPOSITE_ENERGY_THRESHOLD
        + de_dataset["pulse_width"].data[ssd_indices][
            energy >= UltraConstants.COMPOSITE_ENERGY_THRESHOLD
        ]
    )
    composite_energy[energy < UltraConstants.COMPOSITE_ENERGY_THRESHOLD] = energy[
        energy < UltraConstants.COMPOSITE_ENERGY_THRESHOLD
    ]

    energy_norm = get_energy_norm(ssd, composite_energy, ancillary_files)

    return energy_norm


def get_ctof(
    tof: np.ndarray, path_length: np.ndarray, type: str
) -> tuple[NDArray, NDArray]:
    """
    Calculate the corrected TOF and the magnitude of the particle velocity.

    The corrected TOF (ctof) is the TOF normalized with respect
    to a fixed distance dmin between the front and back detectors.
    The normalized TOF is termed the corrected TOF (ctof).
    Further description is available on pages 42-44 of
    IMAP-Ultra Flight Software Specification document.

    Parameters
    ----------
    tof : np.ndarray
        Time of flight (tenths of a nanosecond).
    path_length : np.ndarray
        Path length (r) (hundredths of a millimeter).
    type : str
        Type of event, either "PH" or "SSD".

    Returns
    -------
    ctof : np.ndarray
        Corrected TOF (tenths of a ns).
    magnitude_v : np.ndarray
        Magnitude of the particle velocity (km/s).
    """
    dmin_ctof = getattr(UltraConstants, f"DMIN_{type}_CTOF")

    # Multiply times 100 to convert to hundredths of a millimeter.
    ctof = tof * dmin_ctof * 100 / path_length
    magnitude_v = np.full(len(ctof), -1.0e31, dtype=np.float32)

    # Convert from mm/0.1ns to km/s for valid ctof values
    valid_mask = ctof >= 0
    magnitude_v[valid_mask] = dmin_ctof / ctof[valid_mask] * 1e4

    return ctof, magnitude_v


def determine_species(e_bin: np.ndarray, type: str) -> NDArray:
    """
    Determine the species for pulse-height events.

    Species is determined using the computed e_bin.

    Parameters
    ----------
    e_bin : np.ndarray
        Computed e_bin.
    type : str
        Type of data (PH or SSD).

    Returns
    -------
    species_bin : np.array
        Species bin.
    """
    if type == "PH":
        species_groups = UltraConstants.TOFXPH_SPECIES_GROUPS
    if type == "SSD":
        species_groups = UltraConstants.TOFXE_SPECIES_GROUPS

    non_proton_bins = species_groups["non_proton"]
    proton_bins = species_groups["proton"]

    species_bin = np.full(e_bin.shape, fill_value=2, dtype=int)
    species_bin[np.isin(e_bin, non_proton_bins)] = 0
    species_bin[np.isin(e_bin, proton_bins)] = 1

    return species_bin


def get_phi_theta(
    front_position: tuple, back_position: tuple, d: np.ndarray
) -> tuple[NDArray, NDArray]:
    """
    Compute the instrument angles with range -90 -> 90 degrees.

    Further description is available on page 18 of
    the Ultra Algorithm Theoretical Basis Document.

    Parameters
    ----------
    front_position : tuple of floats
        Front position (xf,yf) (hundredths of a millimeter).
    back_position : tuple of floats
        Back position (xb,yb) (hundredths of a millimeter).
    d : np.ndarray
        Distance from slit to foil (hundredths of a millimeter).

    Returns
    -------
    phi : np.array
        Ultra instrument frame event azimuth.
    theta : np.array
        Ultra instrument frame event elevation.
    """
    path_length = get_path_length(front_position, back_position, d)

    phi = np.arctan((front_position[1] - back_position[1]) / d)
    theta = np.arcsin((front_position[0] - back_position[0]) / path_length)

    return np.degrees(phi), np.degrees(theta)


def get_spin_number(de_met: NDArray, de_spin: NDArray) -> NDArray:
    """
    Get the spin number.

    Parameters
    ----------
    de_met : NDArray
        Mission elapsed time.
    de_spin : NDArray
        Spin number 0-255.

    Returns
    -------
    assigned_spin_number : NDArray
        Spin number for DE data product.
    """
    # DE packet data.
    # Since the spin number in the direct events packet
    # is only 8 bits it goes from 0-255.
    # Within a pointing that means we will always have duplicate spin numbers.
    # In other words, different spins will be represented by the same spin number.
    # Just to make certain that we won't accidentally combine
    # multiple spins we need to sort by time here.
    sort_idx = np.argsort(de_met)
    de_met_sorted = de_met[sort_idx]
    de_spin_sorted = de_spin[sort_idx]
    # Here we are finding the start and end indices of each spin in the sorted array.
    is_new_spin = np.concatenate([[True], de_spin_sorted[1:] != de_spin_sorted[:-1]])
    spin_start_indices = np.where(is_new_spin)[0]
    spin_end_indices = np.append(spin_start_indices[1:], len(de_met_sorted))

    # Universal Spin Table.
    spin_df = get_spin_data()
    # Retrieve the met values of the start of the spin.
    spin_start_mets = spin_df["spin_start_met"].values
    # Retrieve the corresponding spin numbers.
    spin_numbers = spin_df["spin_number"].values
    assigned_spin_number_sorted = np.empty(de_spin_sorted.shape, dtype=np.uint32)
    # These last 8 bits are the same as the spin number in the DE packet.
    # So this will give us choices of which spins are
    # available to assign to the DE data.
    possible_spins = spin_numbers & 0xFF

    # Assign each group based on time.
    for start, end in zip(spin_start_indices, spin_end_indices, strict=False):
        # Now that we have the possible spins from the Universal Spin Table,
        # we match the times of those spins to the nearest times in the DE data.
        possible_times = spin_start_mets[possible_spins == de_spin_sorted[start]]
        # Get nearest time for matching spins.
        nearest_idx = np.abs(possible_times - de_met_sorted[start]).argmin()
        nearest_value = possible_times[nearest_idx]
        assigned_spin_number_sorted[start:end] = spin_numbers[
            spin_start_mets == nearest_value
        ]

    # Undo the sort to match original order.
    assigned_spin_number = np.empty_like(assigned_spin_number_sorted)
    assigned_spin_number[sort_idx] = assigned_spin_number_sorted

    return assigned_spin_number


def get_eventtimes(
    spin: NDArray, phase_angle: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Get the event times.

    Parameters
    ----------
    spin : np.ndarray
        Spin number.
    phase_angle : np.ndarray
        Phase angle.

    Returns
    -------
    event_times : np.ndarray
        Event times in et.
    spin_starts : np.ndarray
        Spin start times in et.
    spin_period_sec : np.ndarray
        Spin period in seconds.

    Notes
    -----
    Equation for event time:
    t = t_(spin start) + t_(spin start sub)/1e6 +
    t_spin_period_sec * phase_angle/720
    """
    spin_df = get_spin_data()
    index = np.searchsorted(spin_df["spin_number"].values, spin)
    spin_starts = (
        spin_df["spin_start_sec_sclk"].values[index]
        + spin_df["spin_start_subsec_sclk"].values[index] / 1e6
    )

    spin_period_sec = spin_df["spin_period_sec"].values[index]

    event_times = spin_starts + spin_period_sec * (phase_angle / 720)

    return sct_to_et(event_times), sct_to_et(spin_starts), spin_period_sec


def interpolate_fwhm(
    lookup_table: pandas.DataFrame,
    energy: NDArray,
    phi_inst: NDArray,
    theta_inst: NDArray,
) -> tuple[NDArray, NDArray]:
    """
    Interpolate phi and theta FWHM values using lookup tables.

    Parameters
    ----------
    lookup_table : DataFrame
        Angular profile lookup table for a given side and sensor.
    energy : NDArray
        Energy values.
    phi_inst : NDArray
        Instrument-frame azimuth angles.
    theta_inst : NDArray
        Instrument-frame elevation angles.

    Returns
    -------
    phi_interp : NDArray
        Interpolated phi FWHM.
    theta_interp : NDArray
        Interpolated theta FWHM.
    """
    interp_phi = LinearNDInterpolator(
        lookup_table[["Energy", "phi_degrees"]].values, lookup_table["phi_fwhm"].values
    )

    interp_theta = LinearNDInterpolator(
        lookup_table[["Energy", "theta_degrees"]].values,
        lookup_table["theta_fwhm"].values,
    )

    # Note: will return nan for those out-of-bounds inputs.
    phi_vals = interp_phi((energy, phi_inst))
    theta_vals = interp_theta((energy, theta_inst))

    phi_interp = np.where(np.isnan(phi_vals), FILLVAL_FLOAT32, phi_vals)
    theta_interp = np.where(np.isnan(theta_vals), FILLVAL_FLOAT32, theta_vals)

    return phi_interp, theta_interp


def get_fwhm(
    start_type: NDArray,
    sensor: str,
    energy: NDArray,
    phi_inst: NDArray,
    theta_inst: NDArray,
    ancillary_files: dict,
) -> tuple[NDArray, NDArray]:
    """
    Interpolate phi and theta FWHM values for each event based on start type.

    Parameters
    ----------
    start_type : NDArray
        Start Type: 1=Left, 2=Right.
    sensor : str
        Sensor name: "ultra45" or "ultra90".
    energy : NDArray
        Energy values for each event.
    phi_inst : NDArray
        Instrument-frame azimuth angle for each event.
    theta_inst : NDArray
        Instrument-frame elevation angle for each event.
    ancillary_files : dict
        Ancillary files containing lookup tables for angular profiles.

    Returns
    -------
    phi_interp : NDArray
        Interpolated phi FWHM values.
    theta_interp : NDArray
        Interpolated theta FWHM values.
    """
    phi_interp = np.full_like(phi_inst, FILLVAL_FLOAT64, dtype=np.float64)
    theta_interp = np.full_like(theta_inst, FILLVAL_FLOAT64, dtype=np.float64)
    lt_table = get_angular_profiles("left", sensor, ancillary_files)
    rt_table = get_angular_profiles("right", sensor, ancillary_files)

    # Left start type
    idx_left = start_type == StartType.Left.value
    phi_interp[idx_left], theta_interp[idx_left] = interpolate_fwhm(
        lt_table, energy[idx_left], phi_inst[idx_left], theta_inst[idx_left]
    )

    # Right start type
    idx_right = start_type == StartType.Right.value
    phi_interp[idx_right], theta_interp[idx_right] = interpolate_fwhm(
        rt_table, energy[idx_right], phi_inst[idx_right], theta_inst[idx_right]
    )

    return phi_interp, theta_interp


def get_efficiency_interpolator(ancillary_files: dict) -> RegularGridInterpolator:
    """
    Return a callable function that interpolates efficiency values for each event.

    Parameters
    ----------
    ancillary_files : dict
        Ancillary files.

    Returns
    -------
    efficiency : NDArray
        Interpolated efficiency values.
    """
    lookup_table = get_energy_efficiencies(ancillary_files)

    theta_vals = np.sort(lookup_table["theta (deg)"].unique())
    phi_vals = np.sort(lookup_table["phi (deg)"].unique())
    energy_column_names = lookup_table.columns[2:].tolist()
    energy_vals = [float(col.replace("keV", "")) for col in energy_column_names]
    efficiency_2d = lookup_table[energy_column_names].values

    efficiency_grid = efficiency_2d.reshape(
        (len(theta_vals), len(phi_vals), len(energy_vals))
    )

    interpolator = RegularGridInterpolator(
        (theta_vals, phi_vals, energy_vals),
        efficiency_grid,
        bounds_error=False,
        fill_value=FILLVAL_FLOAT32,
    )

    return interpolator


def get_efficiency(
    energy: NDArray,
    phi_inst: NDArray,
    theta_inst: NDArray,
    ancillary_files: dict,
    interpolator: RegularGridInterpolator = None,
) -> np.ndarray:
    """
    Return interpolated efficiency values for each event.

    Parameters
    ----------
    energy : NDArray
        Energy values for each event.
    phi_inst : NDArray
        Instrument-frame azimuth angle for each event.
    theta_inst : NDArray
        Instrument-frame elevation angle for each event.
    ancillary_files : dict
        Ancillary files.
    interpolator : RegularGridInterpolator, optional
        Precomputed interpolator to use for efficiency lookup.
        If None, a new interpolator will be created from the ancillary files.

    Returns
    -------
    efficiency : NDArray
        Interpolated efficiency values.
    """
    if not interpolator:
        interpolator = get_efficiency_interpolator(ancillary_files)

    return interpolator((theta_inst, phi_inst, energy))


def determine_ebin_pulse_height(
    energy: NDArray,
    tof: NDArray,
    path_length: NDArray,
    backtofvalid: NDArray,
    coinphvalid: NDArray,
    ancillary_files: dict,
) -> NDArray:
    """
    Determine the species for pulse-height events.

    Species is determined from the particle energy and velocity.
    For velocity, the particle TOF is normalized with respect
    to a fixed distance dmin between the front and back detectors.
    The normalized TOF is termed the corrected TOF (ctof).
    Particle species are determined from
    the energy and ctof using a lookup table.

    Further description is available on pages 42-44 of
    IMAP-Ultra Flight Software Specification document
    (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    energy : NDArray
        Energy from the PH event (keV).
    tof : NDArray
        Time of flight of the PH event (tenths of a nanosecond).
    path_length : NDArray
        Path length (r) (hundredths of a millimeter).
    backtofvalid : NDArray
        Boolean array indicating if the back TOF is valid.
    coinphvalid : NDArray
        Boolean array indicating if the Coincidence PH is valid.
    ancillary_files : dict
        Ancillary files containing the lookup tables.

    Returns
    -------
    bin : np.array
        Species bin.
    """
    # PH event TOF normalization to Z axis
    ctof, _ = get_ctof(tof, path_length, type="PH")

    ebins = np.full(path_length.shape, FILLVAL_UINT8, dtype=np.uint8)
    valid = backtofvalid & coinphvalid
    ebins[valid] = get_ebins(
        "l1b-tofxph", energy[valid], ctof[valid], ebins[valid], ancillary_files
    )

    return ebins


def determine_ebin_ssd(
    energy: NDArray,
    tof: NDArray,
    path_length: NDArray,
    sensor: str,
    ancillary_files: dict,
) -> NDArray:
    """
    Determine the species for SSD events.

    Species is determined from the particle's energy and velocity.
    For velocity, the particle's TOF is normalized with respect
    to a fixed distance dmin between the front and back detectors.
    For SSD events, an adjustment is also made to the path length
    to account for the shorter distances that such events
    travel to reach the detector. The normalized TOF is termed
    the corrected tof (ctof). Particle species are determined from
    the energy and cTOF using a lookup table.

    Further description is available on pages 42-44 of
    IMAP-Ultra Flight Software Specification document
    (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    energy : NDArray
        Energy from the SSD event (keV).
    tof : NDArray
        Time of flight of the SSD event (tenths of a nanosecond).
    path_length : NDArray
        Path length (r) (hundredths of a millimeter).
    sensor : str
        Sensor name: "ultra45" or "ultra90".
    ancillary_files : dict
        Ancillary files containing the lookup tables.

    Returns
    -------
    bin : NDArray
        Species bin.
    """
    # SSD event TOF normalization to Z axis
    ctof, _ = get_ctof(tof, path_length, type="SSD")

    ebins = np.full(path_length.shape, FILLVAL_UINT8, dtype=np.uint8)
    steep_path_length = get_image_params("PathSteepThresh", sensor, ancillary_files)
    medium_path_length = get_image_params("PathMediumThresh", sensor, ancillary_files)

    steep_mask = path_length < steep_path_length
    medium_mask = (path_length >= steep_path_length) & (
        path_length < medium_path_length
    )
    flat_mask = path_length >= medium_path_length

    ebins[steep_mask] = get_ebins(
        f"l1b-{sensor[5::]}sensor-tofxesteep",
        energy[steep_mask],
        ctof[steep_mask],
        ebins[steep_mask],
        ancillary_files,
    )
    ebins[medium_mask] = get_ebins(
        f"l1b-{sensor[5::]}sensor-tofxemedium",
        energy[medium_mask],
        ctof[medium_mask],
        ebins[medium_mask],
        ancillary_files,
    )
    ebins[flat_mask] = get_ebins(
        f"l1b-{sensor[5::]}sensor-tofxeflat",
        energy[flat_mask],
        ctof[flat_mask],
        ebins[flat_mask],
        ancillary_files,
    )

    return ebins


def is_back_tof_valid(
    de_dataset: xarray.Dataset,
    xf: NDArray,
    sensor: str,
    ancillary_files: dict,
) -> NDArray:
    """
    Determine whether back TOF is valid based on stop type.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        Data in xarray format.
    xf : NDArray
        X front position in (hundredths of a millimeter).
        Has same length as de_dataset.
    sensor : str
        Sensor name: "ultra45" or "ultra90".
    ancillary_files : dict
        Ancillary files for lookup.

    Returns
    -------
    valid_mask : NDArray
        Boolean array indicating whether back TOF is valid.

    Notes
    -----
    From page 33 of the IMAP-Ultra Flight Software Specification document.
    """
    _, _, _, _, tofx, tofy = get_ph_tof_and_back_positions(
        de_dataset, xf, "ultra45", ancillary_files
    )
    diff = tofy - tofx

    indices = np.nonzero(
        np.isin(de_dataset["stop_type"], [StopType.Top.value, StopType.Bottom.value])
    )[0]
    de_ph = de_dataset.isel(epoch=indices)

    top_mask = de_ph["stop_type"] == StopType.Top.value
    bottom_mask = de_ph["stop_type"] == StopType.Bottom.value

    valid = np.zeros_like(diff, dtype=bool)

    diff_tp_min = get_image_params("TOFDiffTpMin", sensor, ancillary_files)
    diff_tp_max = get_image_params("TOFDiffTpMax", sensor, ancillary_files)
    diff_bt_min = get_image_params("TOFDiffBtMin", sensor, ancillary_files)
    diff_bt_max = get_image_params("TOFDiffBtMax", sensor, ancillary_files)

    valid[top_mask] = (diff[top_mask] >= diff_tp_min) & (diff[top_mask] <= diff_tp_max)
    valid[bottom_mask] = (diff[bottom_mask] >= diff_bt_min) & (
        diff[bottom_mask] <= diff_bt_max
    )

    return valid


def is_coin_ph_valid(
    etof: NDArray,
    xc: NDArray,
    xb: NDArray,
    stop_north_tdc: NDArray,
    stop_south_tdc: NDArray,
    stop_east_tdc: NDArray,
    stop_west_tdc: NDArray,
    sensor: str,
    ancillary_files: dict,
    quality_flags: NDArray,
) -> NDArray:
    """
    Determine event validity.

    Parameters
    ----------
    etof : NDArray
        Time for the electrons to travel back to the coincidence
        anode (tenths of a nanosecond).
    xc : NDArray
        X coincidence position (hundredths of a millimeter).
    xb : NDArray
        Back positions in x direction (hundredths of a millimeter).
    stop_north_tdc : NDArray
        Stop North Time to Digital Converter.
    stop_south_tdc : NDArray
        Stop South Time to Digital Converter.
    stop_east_tdc : NDArray
        Stop East Time to Digital Converter.
    stop_west_tdc : NDArray
        Stop West Time to Digital Converter.
    sensor : str
        Sensor name: "ultra45" or "ultra90".
    ancillary_files : dict
        Ancillary files for lookup.
    quality_flags : NDArray
        Quality flag to set when there is an outlier.

    Returns
    -------
    combined_mask : NDArray
        Boolean array indicating whether back TOF is valid.

    Notes
    -----
    From page 36 of the IMAP-Ultra Flight Software Specification document.
    """
    # Make certain etof is within range for tenths of a nanosecond.
    etof_valid = (etof >= UltraConstants.ETOFMIN_EVENTFILTER) & (
        etof <= UltraConstants.ETOFMAX_EVENTFILTER
    )

    # Hundredths of a mm.
    diff_x = xc - xb

    t1 = (
        (etof - UltraConstants.ETOFOFF1_EVENTFILTER)
        * UltraConstants.ETOFSLOPE1_EVENTFILTER
        / 1024
    )
    t2 = (
        (etof - UltraConstants.ETOFOFF2_EVENTFILTER)
        * UltraConstants.ETOFSLOPE2_EVENTFILTER
        / 1024
    )

    condition_1 = (diff_x >= t1) & (diff_x <= t2)
    condition_2 = (diff_x >= -t2) & (diff_x <= -t1)

    spatial_valid = condition_1 | condition_2

    sp_n_norm = get_norm(stop_north_tdc, "SpN", sensor, ancillary_files)
    sp_s_norm = get_norm(stop_south_tdc, "SpS", sensor, ancillary_files)
    sp_e_norm = get_norm(stop_east_tdc, "SpE", sensor, ancillary_files)
    sp_w_norm = get_norm(stop_west_tdc, "SpW", sensor, ancillary_files)

    tofx = sp_n_norm + sp_s_norm
    tofy = sp_e_norm + sp_w_norm

    # Units in tenths of a nanosecond
    delta_tof = tofy - tofx

    delta_tof_mask = (delta_tof >= UltraConstants.TOFDIFFTPMIN_EVENTFILTER) & (
        delta_tof <= UltraConstants.TOFDIFFTPMAX_EVENTFILTER
    )

    combined_mask = etof_valid & spatial_valid & delta_tof_mask

    quality_flags[~combined_mask] |= ImapDEOutliersUltraFlags.COINPH.value

    return combined_mask

"""Calculates Extended Raw Events for ULTRA L1b."""

# TODO: Come back and add in FSW logic.
import logging
from enum import Enum

import numpy as np
import pandas
import xarray
from numpy import ndarray
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

from imap_processing.spice.spin import get_spin_data
from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1b.lookup_utils import (
    get_angular_profiles,
    get_back_position,
    get_energy_efficiencies,
    get_energy_norm,
    get_image_params,
    get_norm,
    get_y_adjust,
)

logger = logging.getLogger(__name__)


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


def get_front_x_position(
    start_type: ndarray, start_position_tdc: ndarray, sensor: str
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

    Returns
    -------
    xf : ndarray
        X front position (hundredths of a millimeter).
    """
    # Left and right start types.
    indices = np.nonzero((start_type == 1) | (start_type == 2))

    xftsc = get_image_params("XFTSC", sensor)
    xft_lt_off = get_image_params("XFTLTOFF", sensor)
    xft_rt_off = get_image_params("XFTRTOFF", sensor)
    xft_off = np.where(start_type[indices] == 1, xft_lt_off, xft_rt_off)

    # Calculate xf and convert to hundredths of a millimeter
    xf: ndarray = (xftsc * -start_position_tdc[indices] + xft_off) * 100

    return xf


def get_front_y_position(start_type: ndarray, yb: ndarray) -> tuple[ndarray, ndarray]:
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
    y_adjust_left = get_y_adjust(dy_lut_left) / 100
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
    y_adjust_right = get_y_adjust(dy_lut_right) / 100
    # hundredths of a millimeter
    yf[index_right] = (UltraConstants.YF_ESTIMATE_RIGHT + y_adjust_right) * 100
    # distance adjustment in mm
    distance_adjust_right = np.sqrt(2) * UltraConstants.D_SLIT_FOIL - y_adjust_right
    # hundredths of a millimeter
    d[index_right] = (UltraConstants.SLIT_Z - distance_adjust_right) * 100

    return np.array(d), np.array(yf)


def get_ph_tof_and_back_positions(
    de_dataset: xarray.Dataset, xf: np.ndarray, sensor: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    """
    indices = np.nonzero(
        np.isin(de_dataset["stop_type"], [StopType.Top.value, StopType.Bottom.value])
    )[0]
    de_filtered = de_dataset.isel(epoch=indices)

    xf_ph = xf[indices]

    # There are mismatches between the stop TDCs, i.e., SpN, SpS, SpE, and SpW.
    # This normalizes the TDCs
    sp_n_norm = get_norm(de_filtered["stop_north_tdc"].data, "SpN", sensor)
    sp_s_norm = get_norm(de_filtered["stop_south_tdc"].data, "SpS", sensor)
    sp_e_norm = get_norm(de_filtered["stop_east_tdc"].data, "SpE", sensor)
    sp_w_norm = get_norm(de_filtered["stop_west_tdc"].data, "SpW", sensor)

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
    xb[stop_type_top] = get_back_position(xb_index[stop_type_top], "XBkTp", sensor)
    yb[stop_type_top] = get_back_position(yb_index[stop_type_top], "YBkTp", sensor)

    # Correction for the propagation delay of the start anode and other effects.
    t2[stop_type_top] = get_image_params("TOFSC", sensor) * t1[
        stop_type_top
    ] + get_image_params("TOFTPOFF", sensor)
    # Variable xf_ph divided by 10 to convert to mm.
    tof[stop_type_top] = t2[stop_type_top] + xf_ph[
        stop_type_top
    ] / 10 * get_image_params("XFTTOF", sensor)

    stop_type_bottom = de_filtered["stop_type"].data == StopType.Bottom.value
    xb[stop_type_bottom] = get_back_position(
        xb_index[stop_type_bottom], "XBkBt", sensor
    )
    yb[stop_type_bottom] = get_back_position(
        yb_index[stop_type_bottom], "YBkBt", sensor
    )

    # Correction for the propagation delay of the start anode and other effects.
    t2[stop_type_bottom] = get_image_params("TOFSC", sensor) * t1[
        stop_type_bottom
    ] + get_image_params("TOFBTOFF", sensor)  # 10*ns

    # Variable xf_ph divided by 10 to convert to mm.
    tof[stop_type_bottom] = t2[stop_type_bottom] + xf_ph[
        stop_type_bottom
    ] / 10 * get_image_params("XFTTOF", sensor)

    return tof, t2, xb, yb


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
    de_dataset: xarray.Dataset, sensor: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lookup the Y SSD positions (yb), TOF Offset, and SSD number.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        The input dataset containing STOP_TYPE and SSD_FLAG data.
    sensor : str
        Sensor name.

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
        yb[ssd_flag_mask] = get_image_params(f"YBKSSD{i}", sensor) * 100
        ssd_number[ssd_flag_mask] = i

        tof_offset[
            (de_filtered["start_type"] == StartType.Left.value) & ssd_flag_mask
        ] = get_image_params(f"TOFSSDLTOFF{i}", sensor)
        tof_offset[
            (de_filtered["start_type"] == StartType.Right.value) & ssd_flag_mask
        ] = get_image_params(f"TOFSSDRTOFF{i}", sensor)

    return yb, tof_offset, ssd_number


def calculate_etof_xc(
    de_subset: xarray.Dataset, particle_tof: np.ndarray, sensor: str, location: str
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

    Returns
    -------
    etof : np.ndarray
        Time for the electrons to travel back to the coincidence
        anode (tenths of a nanosecond).
    xc : np.ndarray
        X coincidence position (millimeters).
    """
    # CoinNNorm
    coin_n_norm = get_norm(de_subset["coin_north_tdc"], "CoinN", sensor)
    # CoinSNorm
    coin_s_norm = get_norm(de_subset["coin_south_tdc"], "CoinS", sensor)
    xc = get_image_params(f"XCOIN{location}SC", sensor) * (
        coin_s_norm - coin_n_norm
    ) + get_image_params(f"XCOIN{location}OFF", sensor)  # millimeter

    # Time for the electrons to travel back to coincidence anode.
    t2 = get_image_params("ETOFSC", sensor) * (
        coin_n_norm + coin_s_norm
    ) + get_image_params(f"ETOF{location}OFF", sensor)

    # Multiply by 10 to convert to tenths of a nanosecond.
    etof = t2 * 10 - particle_tof

    return etof, xc


def get_coincidence_positions(
    de_dataset: xarray.Dataset, particle_tof: np.ndarray, sensor: str
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
    etof_top, xc_top = calculate_etof_xc(de_top, particle_tof[index_top], sensor, "TP")
    etof[index_top] = etof_top
    xc_array[index_top] = xc_top

    etof_bottom, xc_bottom = calculate_etof_xc(
        de_bottom, particle_tof[index_bottom], sensor, "BT"
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

    v_x[tof < 0] = np.nan  # used as fillvals
    v_y[tof < 0] = np.nan
    v_z[tof < 0] = np.nan

    velocities = np.vstack((v_x, v_y, v_z)).T

    v_hat = velocities / np.linalg.norm(velocities, axis=1)[:, None]
    r_hat = -v_hat

    return velocities, v_hat, r_hat


def get_ssd_tof(
    de_dataset: xarray.Dataset, xf: np.ndarray, sensor: str
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

    Returns
    -------
    tof : np.ndarray
        Time of flight (tenths of a nanosecond).
    """
    _, tof_offset, ssd_number = get_ssd_back_position_and_tof_offset(de_dataset, sensor)
    indices = np.nonzero(np.isin(de_dataset["stop_type"], [StopType.SSD.value]))[0]

    de_discrete = de_dataset.isel(epoch=indices)["coin_discrete_tdc"]

    time = get_image_params("TOFSSDSC", sensor) * de_discrete.values + tof_offset

    # The scale factor and offsets, and a multiplier to convert xf to a tof offset.
    # Convert xf to mm by dividing by 100.
    tof = (
        time
        + get_image_params("TOFSSDTOTOFF", sensor)
        + xf[indices] / 100 * get_image_params("XFTTOF", sensor)
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

    index_hydrogen = np.where(species == 1)
    energy = np.full_like(v2, np.nan)

    # 1/2 mv^2 in Joules, convert to keV
    energy[index_hydrogen] = (
        0.5 * UltraConstants.MASS_H * v2[index_hydrogen] * UltraConstants.J_KEV
    )

    return energy


def get_energy_pulse_height(
    stop_type: np.ndarray,
    energy: np.ndarray,
    xb: np.ndarray,
    yb: np.ndarray,
    sensor: str,
) -> NDArray[np.float64]:
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

    # Stop type 1
    xlut[indices_top] = (xb[indices_top] / 100 - 25 / 2) * 20 / 50  # mm
    ylut[indices_top] = (yb[indices_top] / 100 + 82 / 2) * 32 / 82  # mm
    # Stop type 2
    xlut[indices_bottom] = (xb[indices_bottom] / 100 + 50 + 25 / 2) * 20 / 50  # mm
    ylut[indices_bottom] = (yb[indices_bottom] / 100 + 82 / 2) * 32 / 82  # mm

    # TODO: waiting on these lookup tables: SpTpPHCorr, SpBtPHCorr
    energy_ph[indices_top] = energy[indices_top] - get_image_params(
        "SPTPPHOFF", sensor
    )  # * SpTpPHCorr[
    # xlut[indices_top], ylut[indices_top]] / 1024

    energy_ph[indices_bottom] = energy[indices_bottom] - get_image_params(
        "SPBTPHOFF", sensor
    )  # * SpBtPHCorr[
    # xlut[indices_bottom], ylut[indices_bottom]] / 1024

    return energy_ph


def get_energy_ssd(de_dataset: xarray.Dataset, ssd: np.ndarray) -> NDArray[np.float64]:
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

    energy_norm = get_energy_norm(ssd, composite_energy)

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


def determine_species(tof: np.ndarray, path_length: np.ndarray, type: str) -> NDArray:
    """
    Determine the species for pulse-height events.

    Species is determined from the particle velocity.
    For velocity, the particle TOF is normalized with respect
    to a fixed distance dmin between the front and back detectors.
    The normalized TOF is termed the corrected TOF (ctof).
    Particle species are determined from ctof using thresholds.

    Further description is available on pages 42-44 of
    IMAP-Ultra Flight Software Specification document.

    Parameters
    ----------
    tof : np.ndarray
        Time of flight of the SSD event (tenths of a nanosecond).
    path_length : np.ndarray
        Path length (r) (hundredths of a millimeter).
    type : str
        Type of data (PH or SSD).

    Returns
    -------
    species_bin : np.array
        Species bin.
    """
    # Event TOF normalization to Z axis
    ctof, _ = get_ctof(tof, path_length, type)
    # Initialize bin array
    species_bin = np.full(len(ctof), 255, dtype=np.uint8)

    # Assign Species 1 ("H") to bins where cTOF is within the specified range
    species_bin[
        (ctof > UltraConstants.CTOF_SPECIES_MIN)
        & (ctof < UltraConstants.CTOF_SPECIES_MAX)
    ] = 1

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
        Event times.
    spin_starts : np.ndarray
        Spin start times.
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

    return event_times, spin_starts, spin_period_sec


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
    phi_interp = interp_phi((energy, phi_inst))
    theta_interp = interp_theta((energy, theta_inst))

    return phi_interp, theta_interp


def get_fwhm(
    start_type: NDArray,
    sensor: str,
    energy: NDArray,
    phi_inst: NDArray,
    theta_inst: NDArray,
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

    Returns
    -------
    phi_interp : NDArray
        Interpolated phi FWHM values.
    theta_interp : NDArray
        Interpolated theta FWHM values.
    """
    phi_interp = np.full_like(phi_inst, np.nan, dtype=np.float64)
    theta_interp = np.full_like(theta_inst, np.nan, dtype=np.float64)
    lt_table = get_angular_profiles("left", sensor)
    rt_table = get_angular_profiles("right", sensor)

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


def get_efficiency(
    energy: NDArray, phi_inst: NDArray, theta_inst: NDArray, ancillary_files: dict
) -> NDArray:
    """
    Interpolate efficiency values for each event.

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
        fill_value=np.nan,
    )

    return interpolator((theta_inst, phi_inst, energy))

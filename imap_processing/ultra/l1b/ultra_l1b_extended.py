"""Calculates Extended Raw Events for ULTRA L1b."""

# TODO: Come back and add in FSW logic.
import logging
from enum import Enum
from typing import ClassVar

import numpy as np
import xarray
from numpy import ndarray
from numpy.typing import NDArray

from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1b.lookup_utils import (
    get_back_position,
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
    PH: ClassVar[list[int]] = [1, 2]
    SSD: ClassVar[list[int]] = [8, 9, 10, 11, 12, 13, 14, 15]


class CoinType(Enum):
    """Coin Type: 1=Top, 2=Bottom."""

    Top = 1
    Bottom = 2


def get_front_x_position(start_type: ndarray, start_position_tdc: ndarray) -> ndarray:
    """
    Calculate the front xf position.

    Converts Start Position Time to Digital Converter (TDC)
    values into units of hundredths of a millimeter using a scale factor and offsets.
    Further description is available on pages 30 of
    IMAP-Ultra Flight Software Specification document (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    start_type : ndarray
        Start Type: 1=Left, 2=Right.
    start_position_tdc : ndarray
        Start Position Time to Digital Converter (TDC).

    Returns
    -------
    xf : ndarray
        X front position (hundredths of a millimeter).
    """
    # Left and right start types.
    indices = np.nonzero((start_type == 1) | (start_type == 2))

    xftsc = get_image_params("XFTSC")
    xft_lt_off = get_image_params("XFTLTOFF")
    xft_rt_off = get_image_params("XFTRTOFF")
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
    IMAP-Ultra Flight Software Specification document
    (7523-9009_Rev_-.pdf).

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
        np.isin(de_dataset["STOP_TYPE"], [StopType.Top.value, StopType.Bottom.value])
    )[0]
    de_filtered = de_dataset.isel(epoch=indices)

    xf_ph = xf[indices]

    # There are mismatches between the stop TDCs, i.e., SpN, SpS, SpE, and SpW.
    # This normalizes the TDCs
    sp_n_norm = get_norm(de_filtered["STOP_NORTH_TDC"].data, "SpN", sensor)
    sp_s_norm = get_norm(de_filtered["STOP_SOUTH_TDC"].data, "SpS", sensor)
    sp_e_norm = get_norm(de_filtered["STOP_EAST_TDC"].data, "SpE", sensor)
    sp_w_norm = get_norm(de_filtered["STOP_WEST_TDC"].data, "SpW", sensor)

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
    stop_type_top = de_filtered["STOP_TYPE"].data == StopType.Top.value
    xb[stop_type_top] = get_back_position(xb_index[stop_type_top], "XBkTp", sensor)
    yb[stop_type_top] = get_back_position(yb_index[stop_type_top], "YBkTp", sensor)

    # Correction for the propagation delay of the start anode and other effects.
    t2[stop_type_top] = get_image_params("TOFSC") * t1[
        stop_type_top
    ] + get_image_params("TOFTPOFF")
    # Variable xf_ph divided by 10 to convert to mm.
    tof[stop_type_top] = t2[stop_type_top] + xf_ph[
        stop_type_top
    ] / 10 * get_image_params("XFTTOF")

    stop_type_bottom = de_filtered["STOP_TYPE"].data == StopType.Bottom.value
    xb[stop_type_bottom] = get_back_position(
        xb_index[stop_type_bottom], "XBkBt", sensor
    )
    yb[stop_type_bottom] = get_back_position(
        yb_index[stop_type_bottom], "YBkBt", sensor
    )

    # Correction for the propagation delay of the start anode and other effects.
    t2[stop_type_bottom] = get_image_params("TOFSC") * t1[
        stop_type_bottom
    ] + get_image_params("TOFBTOFF")  # 10*ns

    # Variable xf_ph divided by 10 to convert to mm.
    tof[stop_type_bottom] = t2[stop_type_bottom] + xf_ph[
        stop_type_bottom
    ] / 10 * get_image_params("XFTTOF")

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
    de_dataset: xarray.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lookup the Y SSD positions (yb), TOF Offset, and SSD number.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        The input dataset containing STOP_TYPE and SSD_FLAG data.

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
    indices = np.nonzero(np.isin(de_dataset["STOP_TYPE"], StopType.SSD.value))[0]
    de_filtered = de_dataset.isel(epoch=indices)

    yb = np.zeros(len(indices), dtype=np.float64)
    ssd_number = np.zeros(len(indices), dtype=int)
    tof_offset = np.zeros(len(indices), dtype=np.float64)

    for i in range(8):
        ssd_flag_mask = de_filtered[f"SSD_FLAG_{i}"].data == 1

        # Multiply ybs times 100 to convert to hundredths of a millimeter.
        yb[ssd_flag_mask] = get_image_params(f"YBKSSD{i}") * 100
        ssd_number[ssd_flag_mask] = i

        tof_offset[
            (de_filtered["START_TYPE"] == StartType.Left.value) & ssd_flag_mask
        ] = get_image_params(f"TOFSSDLTOFF{i}")
        tof_offset[
            (de_filtered["START_TYPE"] == StartType.Right.value) & ssd_flag_mask
        ] = get_image_params(f"TOFSSDRTOFF{i}")

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
    coin_n_norm = get_norm(de_subset["COIN_NORTH_TDC"], "CoinN", sensor)
    # CoinSNorm
    coin_s_norm = get_norm(de_subset["COIN_SOUTH_TDC"], "CoinS", sensor)
    xc = get_image_params(f"XCOIN{location}SC") * (
        coin_s_norm - coin_n_norm
    ) + get_image_params(f"XCOIN{location}OFF")  # millimeter

    # Time for the electrons to travel back to coincidence anode.
    t2 = get_image_params("ETOFSC") * (coin_n_norm + coin_s_norm) + get_image_params(
        f"ETOF{location}OFF"
    )

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
    IMAP-Ultra Flight Software Specification document
    (7523-9009_Rev_-.pdf).

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
    index_top = np.nonzero(np.isin(de_dataset["COIN_TYPE"], CoinType.Top.value))[0]
    de_top = de_dataset.isel(epoch=index_top)

    index_bottom = np.nonzero(np.isin(de_dataset["COIN_TYPE"], CoinType.Bottom.value))[
        0
    ]
    de_bottom = de_dataset.isel(epoch=index_bottom)

    etof = np.zeros(len(de_dataset["COIN_TYPE"]), dtype=np.float64)
    xc_array = np.zeros(len(de_dataset["COIN_TYPE"]), dtype=np.float64)

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


def get_unit_vector(
    front_position: tuple[NDArray, NDArray],
    back_position: tuple[NDArray, NDArray],
    d: np.ndarray,
    tof: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Determine the particle velocity.

    The equation is: velocity = ((xf - xb), (yf - yb), d).

    Further description is available on pages 39 of
    IMAP-Ultra Flight Software Specification document
    (7523-9009_Rev_-.pdf).

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
    vhat_x : np.array
        Normalized component of the velocity vector in x direction.
    vhat_y : np.array
        Normalized component of the velocity vector in y direction.
    vhat_z : np.array
        Normalized component of the velocity vector in z direction.
    """
    if tof[tof < 0].any():
        logger.info("Negative tof values found.")

    delta_x = front_position[0] - back_position[0]
    delta_y = front_position[1] - back_position[1]

    v_x = delta_x / tof
    v_y = delta_y / tof
    v_z = d / tof

    # Magnitude of the velocity vector
    magnitude_v = np.sqrt(v_x**2 + v_y**2 + v_z**2)

    vhat_x = -v_x / magnitude_v
    vhat_y = -v_y / magnitude_v
    vhat_z = -v_z / magnitude_v

    vhat_x[tof < 0] = np.nan  # used as fillvals
    vhat_y[tof < 0] = np.nan
    vhat_z[tof < 0] = np.nan

    return vhat_x, vhat_y, vhat_z


def get_ssd_tof(de_dataset: xarray.Dataset, xf: np.ndarray) -> NDArray[np.float64]:
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
    IMAP-Ultra Flight Software Specification document
    (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    de_dataset : xarray.Dataset
        Data in xarray format.
    xf : np.array
        Front x position (hundredths of a millimeter).

    Returns
    -------
    tof : np.ndarray
        Time of flight (tenths of a nanosecond).
    """
    _, tof_offset, ssd_number = get_ssd_back_position_and_tof_offset(de_dataset)
    indices = np.nonzero(np.isin(de_dataset["STOP_TYPE"], [StopType.SSD.value]))[0]

    de_discrete = de_dataset.isel(epoch=indices)["COIN_DISCRETE_TDC"]

    time = get_image_params("TOFSSDSC") * de_discrete.values + tof_offset

    # The scale factor and offsets, and a multiplier to convert xf to a tof offset.
    # Convert xf to mm by dividing by 100.
    tof = (
        time
        + get_image_params("TOFSSDTOTOFF")
        + xf[indices] / 100 * get_image_params("XFTTOF")
    ) * 10

    # Convert TOF to tenths of a nanosecond.
    return np.asarray(tof, dtype=np.float64)


def get_energy_pulse_height(
    stop_type: np.ndarray, energy: np.ndarray, xb: np.ndarray, yb: np.ndarray
) -> NDArray[np.float64]:
    """
    Calculate the pulse-height energy.

    Calculate energy measured using the
    pulse height from the stop anode.
    Lookup tables (lut) are used for corrections.
    Further description is available on pages 40-41 of
    IMAP-Ultra Flight Software Specification document
    (7523-9009_Rev_-.pdf).

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
        "SPTPPHOFF"
    )  # * SpTpPHCorr[
    # xlut[indices_top], ylut[indices_top]] / 1024

    energy_ph[indices_bottom] = energy[indices_bottom] - get_image_params(
        "SPBTPHOFF"
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
    IMAP-Ultra Flight Software Specification document
    (7523-9009_Rev_-.pdf).

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
    ssd_indices = np.where(de_dataset["STOP_TYPE"].data >= 8)[0]
    energy = de_dataset["ENERGY_PH"].data[ssd_indices]

    composite_energy = np.empty(len(energy), dtype=np.float64)

    composite_energy[energy >= UltraConstants.COMPOSITE_ENERGY_THRESHOLD] = (
        UltraConstants.COMPOSITE_ENERGY_THRESHOLD
        + de_dataset["PULSE_WIDTH"].data[ssd_indices][
            energy >= UltraConstants.COMPOSITE_ENERGY_THRESHOLD
        ]
    )
    composite_energy[energy < UltraConstants.COMPOSITE_ENERGY_THRESHOLD] = energy[
        energy < UltraConstants.COMPOSITE_ENERGY_THRESHOLD
    ]

    energy_norm = get_energy_norm(ssd, composite_energy)

    return energy_norm


def get_ctof(tof: np.ndarray, path_length: np.ndarray, type: str) -> NDArray:
    """
    Calculate the corrected TOF.

    The corrected TOF (ctof) is the TOF normalized with respect
    to a fixed distance dmin between the front and back detectors.
    The normalized TOF is termed the corrected TOF (ctof).
    Further description is available on pages 42-44 of
    IMAP-Ultra Flight Software Specification document
    (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    tof : np.ndarray
        Time of flight (tenths of a nanosecond).
    path_length : np.ndarray
        Path length (r) (hundredths of a millimeter).
    type : str
        Type of event, either "ph" or "ssd".

    Returns
    -------
    ctof : np.ndarray
        Corrected TOF (tenths of a ns).
    """
    dmin_ctof = getattr(UltraConstants, f"DMIN_{type}_CTOF")

    # Multiply times 100 to convert to hundredths of a millimeter.
    ctof = tof * dmin_ctof * 100 / path_length

    return ctof


def determine_species_pulse_height(
    energy: np.ndarray, tof: np.ndarray, path_length: np.ndarray
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
    energy : np.ndarray
        Energy from the SSD event (keV).
    tof : np.ndarray
        Time of flight of the SSD event (tenths of a nanosecond).
    path_length : np.ndarray
        Path length (r) (hundredths of a millimeter).

    Returns
    -------
    bin : np.array
        Species bin.
    """
    # PH event TOF normalization to Z axis
    ctof = get_ctof(tof, path_length, "PH")
    # TODO: need lookup tables
    # placeholder
    bin = np.zeros(len(ctof))
    # bin = PHxTOFSpecies[ctof, energy]

    return bin


def determine_species_ssd(
    energy: np.ndarray, tof: np.ndarray, path_length: np.ndarray
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
    energy : np.ndarray
        Energy from the SSD event (keV).
    tof : np.ndarray
        Time of flight of the SSD event (tenths of a nanosecond).
    path_length : np.ndarray
        Path length (r) (hundredths of a millimeter).

    Returns
    -------
    bin : np.ndarray
        Species bin.
    """
    # SSD event TOF normalization to Z axis
    ctof = get_ctof(tof, path_length, "SSD")

    bin = np.zeros(len(ctof))  # placeholder

    # TODO: get these lookup tables
    # if r < get_image_params("PathSteepThresh"):
    #     # bin = ExTOFSpeciesSteep[energy, ctof]
    # elif r < get_image_params("PathMediumThresh"):
    #     # bin = ExTOFSpeciesMedium[energy, ctof]
    # else:
    #     # bin = ExTOFSpeciesFlat[energy, ctof]

    return bin

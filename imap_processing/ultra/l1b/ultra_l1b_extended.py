"""Calculates Extended Raw Events for ULTRA L1b."""

from enum import Enum
import numpy as np
from numpy import ndarray
import xarray

from imap_processing.ultra.l1b.lookup_utils import (
    get_back_position,
    get_norm,
    get_image_params,
    get_y_adjust,
)

# Constants in IMAP-Ultra Flight Software Specification document.
D_SLIT_FOIL = 3.39  # shortest distance from slit to foil (mm)
SLIT_Z = 44.89  # position of slit on Z axis (mm)
YF_ESTIMATE_LEFT = 40.0  # front position of particle for left shutter (mm)
YF_ESTIMATE_RIGHT = -40  # front position of particle for right shutter (mm)
N_ELEMENTS = 256  # number of elements in lookup table
TRIG_CONSTANT = 81.92  # trigonometric constant (mm)
# TODO: make lookup tables into config files.


class StopType(Enum):
    """Stop Type: 1=Top, 2=Bottom."""
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
        (YF_ESTIMATE_LEFT - yb[index_left] / 100) * N_ELEMENTS / TRIG_CONSTANT + 0.5
    )
    # y adjustment in mm
    y_adjust_left = get_y_adjust(dy_lut_left) / 100
    # hundredths of a millimeter
    yf[index_left] = (YF_ESTIMATE_LEFT - y_adjust_left) * 100
    # distance adjustment in mm
    distance_adjust_left = np.sqrt(2) * D_SLIT_FOIL - y_adjust_left
    # hundredths of a millimeter
    d[index_left] = (SLIT_Z - distance_adjust_left) * 100

    # Compute adjustments for right start type
    dy_lut_right = np.floor(
        (yb[index_right] / 100 - YF_ESTIMATE_RIGHT) * N_ELEMENTS / TRIG_CONSTANT + 0.5
    )
    # y adjustment in mm
    y_adjust_right = get_y_adjust(dy_lut_right) / 100
    # hundredths of a millimeter
    yf[index_right] = (YF_ESTIMATE_RIGHT + y_adjust_right) * 100
    # distance adjustment in mm
    distance_adjust_right = np.sqrt(2) * D_SLIT_FOIL - y_adjust_right
    # hundredths of a millimeter
    d[index_right] = (SLIT_Z - distance_adjust_right) * 100

    return np.array(d), np.array(yf)


def get_ph_tof_and_back_positions(
    de_dataset: xarray.Dataset, xf: np.array, sensor: str
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
    sensor : str
        Sensor name.

    Returns
    -------
    tof : np.array
        Time of flight (tenths of a nanosecond).
    t2 : np.array
        Particle time of flight from start to stop (tenths of a nanosecond).
    xb : np.array
        Back positions in x direction (hundredths of a millimeter).
    yb : np.array
        Back positions in y direction (hundredths of a millimeter).
    """
    indices = np.where(np.isin(de_dataset["STOP_TYPE"], [1, 2]))[0]
    xf = xf[indices]

    # There are mismatches between the stop TDCs, i.e., SpN, SpS, SpE, and SpW.
    # This normalizes the TDCs
    sp_n_norm = get_norm(de_dataset["STOP_NORTH_TDC"].data[indices], "SpN", sensor)
    sp_s_norm = get_norm(de_dataset["STOP_SOUTH_TDC"].data[indices], "SpS", sensor)
    sp_e_norm = get_norm(de_dataset["STOP_EAST_TDC"].data[indices], "SpE", sensor)
    sp_w_norm = get_norm(de_dataset["STOP_WEST_TDC"].data[indices], "SpW", sensor)

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

    xb = np.zeros(len(de_dataset["STOP_TYPE"]))
    yb = np.zeros(len(de_dataset["STOP_TYPE"]))

    # particle_tof (t2) used later to compute etof
    t2 = np.zeros(len(de_dataset["STOP_TYPE"]))
    tof = np.zeros(len(de_dataset["STOP_TYPE"]))

    # Stop Type: 1=Top, 2=Bottom
    # Convert converts normalized TDC values into units of
    # hundredths of a millimeter using lookup tables.
    index_top = indices[de_dataset["STOP_TYPE"].data[indices] == 1]
    stop_type_top = de_dataset["STOP_TYPE"].data[indices] == 1
    xb[index_top] = get_back_position(xb_index[stop_type_top], "XBkTp", sensor)
    yb[index_top] = get_back_position(yb_index[stop_type_top], "YBkTp", sensor)

    # Correction for the propagation delay of the start anode and other effects.
    t2[index_top] = get_image_params("TOFSC") * t1[stop_type_top] + get_image_params(
        "TOFTPOFF"
    )
    tof[index_top] = t2[index_top] + xf[stop_type_top] * get_image_params("XFTTOF")

    index_bottom = indices[de_dataset["STOP_TYPE"].data[indices] == 2]
    stop_type_bottom = de_dataset["STOP_TYPE"].data[indices] == 2
    xb[index_bottom] = get_back_position(xb_index[stop_type_bottom], "XBkBt", sensor)
    yb[index_bottom] = get_back_position(yb_index[stop_type_bottom], "YBkBt", sensor)

    # Correction for the propagation delay of the start anode and other effects.
    t2[index_bottom] = get_image_params("TOFSC") * t1[
        stop_type_bottom
    ] + get_image_params("TOFBTOFF")  # 10*ns
    tof[index_bottom] = t2[index_bottom] + xf[stop_type_bottom] * get_image_params(
        "XFTTOF"
    )
    # Multiply by 100 to get tenths of a nanosecond.
    tof = tof[indices] * 100
    t2 = t2[indices]
    xb = xb[indices]
    yb = yb[indices]

    return tof, t2, xb, yb


def get_path_length(front_position: tuple, back_position: tuple, d: float) -> float:
    """
    Calculate the path length.

    Parameters
    ----------
    front_position : tuple of floats
        Front position (xf,yf) (hundredths of a millimeter).
    back_position : tuple of floats
        Back position (xb,yb) (hundredths of a millimeter).
    d : float
        Distance from slit to foil (hundredths of a millimeter).

    Returns
    -------
    r : float
        Path length (hundredths of a millimeter).
    """
    r: float = np.sqrt(
        (front_position[0] - back_position[0]) ** 2
        + (front_position[1] - back_position[1]) ** 2
        + (d) ** 2
    )

    return r

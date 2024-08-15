"""Calculates Extended Raw Events for ULTRA L1b."""

import numpy as np
from numpy import ndarray

from imap_processing.ultra.l1b.lookup_utils import (
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

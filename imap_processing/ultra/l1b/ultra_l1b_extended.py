"""Calculates Extended Raw Events for ULTRA L1b."""

import numpy as np

from imap_processing.ultra.l1b.lookup_utils import (
    get_image_params,
    get_y_adjust,
)

# TODO: make lookup tables into config files.


def get_front_x_position(
    start_type: np.array, start_position_tdc: np.array
) -> np.array:
    """
    Calculate the front xf position.

    Converts Start Position Time to Digital Converter (TDC)
    values into units of hundredths of a millimeter using a scale factor and offsets.
    Further description is available on pages 30 of
    IMAP-Ultra Flight Software Specification document (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    start_type : np.array
        Start Type: 1=Left, 2=Right.
    start_position_tdc : np.array
        Start Position Time to Digital Converter (TDC).

    Returns
    -------
    xf : np.array
        X front position (hundredths of a millimeter).
    """
    # Left and right start types.
    indices = np.where((start_type == 1) | (start_type == 2))

    xftsc = get_image_params("XFTSC")
    xft_lt_off = get_image_params("XFTLTOFF")
    xft_rt_off = get_image_params("XFTRTOFF")
    xft_off = np.where(start_type[indices] == 1, xft_lt_off, xft_rt_off)

    # Calculate xf and convert to hundredths of a millimeter
    xf = (xftsc * -start_position_tdc[indices] + xft_off) * 100

    return xf


def get_front_y_position(
    start_type: np.array, yb: np.array
) -> tuple[np.array, np.array]:
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
    # df in IMAP-Ultra Flight Software Specification document.
    d_slit_foil = 3.39  # shortest distance from slit to foil (mm)
    # z_ds in IMAP-Ultra Flight Software Specification document.
    slit_z = 44.89  # position of slit on Z axis (mm)

    # Determine start types
    start_type_left = start_type == 1
    start_type_right = start_type == 2
    index_array = np.arange(len(start_type))
    index_left = index_array[start_type_left]
    index_right = index_array[start_type_right]

    yf = np.zeros(len(start_type))
    d = np.zeros(len(start_type))

    # front position of particle for left shutter (mm)
    yf_estimate_left = 40.0
    # front position of particle for right shutter (mm)
    yf_estimate_right = -40.0
    # number of elements in lookup table (pg 38)
    n_elements = 256
    # trigonometric constant (mm) (pg 38)
    trig_constant = 81.92

    # Compute adjustments for left start type
    dy_lut_left = np.floor(
        (yf_estimate_left - yb[start_type_left] / 100) * n_elements / trig_constant
        + 0.5
    )
    # y adjustment in mm
    y_adjust_left = get_y_adjust(dy_lut_left) / 100
    # hundredths of a millimeter
    yf[index_left] = (yf_estimate_left - y_adjust_left) * 100
    # distance adjustment in mm
    distance_adjust_left = np.sqrt(2) * d_slit_foil - y_adjust_left
    # hundredths of a millimeter
    d[index_left] = (slit_z - distance_adjust_left) * 100

    # Compute adjustments for right start type
    dy_lut_right = np.floor(
        (yb[start_type_right] / 100 - yf_estimate_right) * n_elements / trig_constant
        + 0.5
    )
    # y adjustment in mm
    y_adjust_right = get_y_adjust(dy_lut_right) / 100
    # hundredths of a millimeter
    yf[index_right] = (yf_estimate_right + y_adjust_right) * 100
    # distance adjustment in mm
    distance_adjust_right = np.sqrt(2) * d_slit_foil - y_adjust_right
    # hundredths of a millimeter
    d[index_right] = (slit_z - distance_adjust_right) * 100

    return d, yf

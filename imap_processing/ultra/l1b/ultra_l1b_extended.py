"""Calculates Extended Raw Events for ULTRA L1b."""

import logging

import numpy as np
import xarray

from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.ultra.l1b.lookup_utils import (
    get_back_position,
    get_energy_norm,
    get_image_params,
    get_norm,
    get_y_adjust,
)

# TODO: add in logic similar to FSW document.
# TODO: make lookup tables into config files.


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
    IMAP-Ultra Flight Software Specification document.

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

    # Compute adjustments for left start type
    dy_lut_left = np.round((yf_estimate_left - yb[start_type_left] / 100) * 256 / 81.92)
    # y adjustment in mm
    y_adjust_left = get_y_adjust(dy_lut_left) / 100
    # hundredths of a millimeter
    yf[index_left] = (yf_estimate_left - y_adjust_left) * 100
    # distance adjustment in mm
    distance_adjust_left = np.sqrt(2) * d_slit_foil - y_adjust_left
    # hundredths of a millimeter
    d[index_left] = (slit_z - distance_adjust_left) * 100

    # Compute adjustments for right start type
    dy_lut_right = np.round(
        (yb[start_type_right] / 100 - yf_estimate_right) * 256 / 81.92
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
    etof_sorted : np.ndarray
        Time for the electrons to travel back to
        coincidence anode (tenths of a nanosecond).
    xc_sorted : np.ndarray
        X coincidence position (hundredths of a millimeter).
    """
    index_top = np.where(de_dataset["COIN_TYPE"] == 1)[0]
    index_bottom = np.where(de_dataset["COIN_TYPE"] == 2)[0]
    index = np.concatenate((index_top, index_bottom))

    # Normalized TDCs
    # For the stop anode, there are mismatches between the coincidence TDCs,
    # i.e., CoinN and CoinS. They must be normalized via lookup tables.

    # TpCoinNNorm Top
    coin_n_norm_top = get_norm(de_dataset["COIN_NORTH_TDC"][index_top], "CoinN", sensor)
    # TpCoinSNorm Top
    coin_s_norm_top = get_norm(de_dataset["COIN_SOUTH_TDC"][index_top], "CoinS", sensor)
    t1_top = coin_n_norm_top + coin_s_norm_top  # /2 incorporated into scale
    xc_top = get_image_params("XCOINTPSC") * (
        coin_s_norm_top - coin_n_norm_top
    ) + get_image_params("XCOINTPOFF")  # millimeter
    t2_top = get_image_params("ETOFSC") * t1_top + get_image_params("ETOFTPOFF")

    # TpCoinNNorm Bottom
    coin_n_norm_bottom = get_norm(
        de_dataset["COIN_NORTH_TDC"][index_bottom], "CoinN", sensor
    )
    # TpCoinSNorm Bottom
    coin_s_norm_bottom = get_norm(
        de_dataset["COIN_SOUTH_TDC"][index_bottom], "CoinS", sensor
    )
    t1_bottom = coin_n_norm_bottom + coin_s_norm_bottom  # /2 incorporated into scale
    xc_bottom = get_image_params("XCOINBTSC") * (
        coin_s_norm_bottom - coin_n_norm_bottom
    ) + get_image_params("XCOINBTOFF")  # millimeter
    t2_bottom = get_image_params("ETOFSC") * t1_bottom + get_image_params("ETOFBTOFF")

    # Time for the electrons to travel back to coincidence anode.
    t2 = np.concatenate((t2_top, t2_bottom))
    # Convert to hundredths of a millimeter by multiplying times 100
    xc = np.concatenate((xc_top * 100, xc_bottom * 100))

    etof = t2 - particle_tof

    etof_sorted = etof[np.argsort(index)]
    xc_sorted = xc[np.argsort(index)]

    return etof_sorted, xc_sorted


def get_ssd_offset_and_positions(
    de_dataset: xarray.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Figure out what SSD a particle hit.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        Data in xarray format.

    Returns
    -------
    yb_sorted : np.array
        Y ssd position (hundredths of a millimeter).
    tof_offsets_sorted : np.array
        Time of flight offset (tenths of a nanosecond).
    ssds_sorted : np.array
        SSD number.
    """
    ssd_indices = np.array([], dtype=int)
    ybs = np.array([], dtype=np.float64)
    tof_offsets = np.array([], dtype=np.float64)
    ssds = np.array([], dtype=int)

    # START_TYPE: 1=Left
    indices = np.where(
        (de_dataset["STOP_TYPE"] >= 8) & (de_dataset["START_TYPE"] == 1)
    )[0]
    for i in range(8):
        ssd_index = indices[de_dataset[f"SSD_FLAG_{i}"].data[indices] == 1]
        ssd_indices = np.concatenate((ssd_indices, ssd_index))

        ssds = np.concatenate((ssds, np.full(len(ssd_index), i, dtype=int)))

        yb = np.full(len(ssd_index), get_image_params(f"YBKSSD{i}"))
        ybs = np.concatenate((ybs, yb))

        tof_offset = np.full(len(ssd_index), get_image_params(f"TOFSSDLTOFF{i}"))
        tof_offsets = np.concatenate((tof_offsets, tof_offset))

    # START_TYPE: 2=Right
    indices = np.where(
        (de_dataset["STOP_TYPE"] >= 8) & (de_dataset["START_TYPE"] == 2)
    )[0]
    for i in range(8):
        ssd_index = indices[de_dataset[f"SSD_FLAG_{i}"].data[indices] == 1]
        ssd_indices = np.concatenate((ssd_indices, ssd_index))

        ssds = np.concatenate((ssds, np.full(len(ssd_index), i, dtype=int)))

        yb = np.full(len(ssd_index), get_image_params(f"YBKSSD{i}"))
        ybs = np.concatenate((ybs, yb))

        tof_offset = np.full(len(ssd_index), get_image_params(f"TOFSSDRTOFF{i}"))
        tof_offsets = np.concatenate((tof_offsets, tof_offset))

    # multiply ybs times 100 to convert to hundredths of a millimeter.
    yb_sorted = ybs[np.argsort(ssd_indices)] * 100
    tof_offsets_sorted = tof_offsets[np.argsort(ssd_indices)]
    ssds_sorted = ssds[np.argsort(ssd_indices)]

    return yb_sorted, tof_offsets_sorted, ssds_sorted


def get_ssd_tof(
    de_dataset: xarray.Dataset, xf: np.array
) -> tuple[np.ndarray, np.ndarray]:
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
    ssd : np.ndarray
        SSD number.
    """
    _, tof_offsets, ssd = get_ssd_offset_and_positions(de_dataset)
    ssd_indices = np.where(de_dataset["STOP_TYPE"] >= 8)[0]

    time = (
        get_image_params("TOFSSDSC") * de_dataset["COIN_DISCRETE_TDC"].data[ssd_indices]
        + tof_offsets
    )

    # The scale factor and offsets, and a multiplier to convert xf to a tof offset.
    tof = (
        time
        + get_image_params("TOFSSDTOTOFF")
        + xf[ssd_indices] * get_image_params("XFTTOF")
    )

    return tof, ssd


def get_energy_pulse_height(
    stop_type: np.array, xb: np.array, yb: np.array
) -> np.array:
    """
    Calculate the pulse-height energy.

    Calculate the energy measured using the
    pulse height from the stop anode.
    Lookup tables (lut) are used for corrections.
    Further description is available on pages 40-41 of
    IMAP-Ultra Flight Software Specification document
    (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    stop_type : np.array
        Stop type: 1=Top, 2=Bottom.
    xb : np.array
        X back position (hundredths of a millimeter).
    yb : np.array
        Y back position (hundredths of a millimeter).

    Returns
    -------
    energy : np.array
        Energy measured using the pulse height
        from the stop anode (DN).
    """
    indices_top = np.where(stop_type == 1)[0]
    indices_bottom = np.where(stop_type == 2)[0]

    # Stop type 1
    xlut = (xb[indices_top] / 100 - 25 / 2) * 20 / 50  # mm
    ylut = (yb[indices_top] / 100 + 82 / 2) * 32 / 82  # mm
    energy_1 = xlut + ylut  # placeholder
    # energy = de_dataset["ENERGY_PH"].data[indices_1] -
    # get_image_params("SpTpPHOffset")
    # TODO * SpTpPHCorr[xlut, ylut] / 1024; George Clark working on it

    # Stop type 2
    xlut = (xb[indices_bottom] / 100 + 50 + 25 / 2) * 20 / 50  # mm
    ylut = (yb[indices_bottom] / 100 + 82 / 2) * 32 / 82  # mm
    energy_2 = xlut + ylut  # placeholder
    # energy = de_dataset["ENERGY_PH"].data[indices_2] -
    # energy = pulse_height - get_image_params("SpBtPHOffset")
    # TODO * SpBtPHCorr[xlut, ylut]/1024; George Clark working on it

    energy = np.concatenate((energy_1, energy_2))

    return energy


def get_energy_ssd(de_dataset: xarray.Dataset, ssd: np.array) -> np.ndarray:
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
    ssd : np.array
        SSD number.

    Returns
    -------
    energy : np.ndarray
        Energy measured using the SSD.
    """
    # TODO: find a reference for this
    composite_energy_threshold = 1706

    ssd_indices = np.where(de_dataset["STOP_TYPE"] >= 8)[0]
    energy = de_dataset["ENERGY_PH"].data[ssd_indices]

    composite_energy = np.empty(len(energy), dtype=np.float64)

    composite_energy[energy >= composite_energy_threshold] = (
        composite_energy_threshold
        + de_dataset["PULSE_WIDTH"][ssd_indices][energy >= composite_energy_threshold]
    )
    composite_energy[energy < composite_energy_threshold] = energy[
        energy < composite_energy_threshold
    ]

    energy_norm = get_energy_norm(ssd, composite_energy)

    return energy_norm


def determine_species_ssd(
    energy: np.array, tof: np.array, r: np.array
) -> tuple[np.array, np.array]:
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
    energy : np.array
        Energy from the SSD event (keV).
    tof : np.array
        Time of flight of the SSD event (tenths of a nanosecond).
    r : np.array
        Path length (hundredths of a millimeter).

    Returns
    -------
    ctof : np.array
        Corrected TOF.
    bin : np.array
        Species bin.
    """
    z_dstop = 2.6 / 2  # position of stop foil on Z axis (mm)
    z_ds = 46.19 - z_dstop  # position of slit on Z axis (mm)
    df = 3.39  # distance from slit to foil (mm)

    # PH event TOF normalization to Z axis
    # Note: there is a type in IMAP-Ultra Flight
    # Software Specification document
    dmin = z_ds - np.sqrt(2) * df  # (mm)
    dmin_ssd_ctof = dmin**2 / (dmin - z_dstop)  # (mm)
    ctof = tof * dmin_ssd_ctof / (r / 100)

    bin = np.zeros(len(ctof))  # placeholder

    # TODO: get these lookup tables
    # if r < get_image_params("PathSteepThresh"):
    #     # bin = ExTOFSpeciesSteep[energy, ctof]
    # elif r < get_image_params("PathMediumThresh"):
    #     # bin = ExTOFSpeciesMedium[energy, ctof]
    # else:
    #     # bin = ExTOFSpeciesFlat[energy, ctof]

    # Convert ctof from nanoseconds to tenths of a nanosecond
    return ctof * 10, bin


def determine_species_pulse_height(
    energy: np.array, tof: np.array, r: np.array
) -> tuple[np.array, np.array]:
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
    energy : np.array
        Energy from the SSD event (keV).
    tof : np.array
        Time of flight of the SSD event (tenths of a nanosecond).
    r : np.array
        Path length (hundredths of a millimeter).

    Returns
    -------
    ctof : np.array
        Corrected TOF.
    bin : np.array
        Species bin.
    """
    z_dstop = 2.6 / 2  # position of stop foil on Z axis (mm)
    z_ds = 46.19 - z_dstop  # position of slit on Z axis (mm)
    df = 3.39  # distance from slit to foil (mm)

    # PH event TOF normalization to Z axis
    # Note: there is a type in IMAP-Ultra Flight
    # Software Specification document
    dmin = z_ds - np.sqrt(2) * df  # (mm)

    ctof = tof * dmin / r
    # TODO: need lookup tables
    # placeholder
    bin = np.zeros(len(ctof))
    # bin = PHxTOFSpecies[ctof, energy]

    return ctof, bin


def get_particle_velocity(
    front_position: tuple, back_position: tuple, d: np.array, tof: np.array
) -> tuple[np.array, np.array, np.array]:
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
        logging.info("Negative tof values found.")

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

    vhat_x[tof < 0] = GlobalConstants.INT_FILLVAL
    vhat_y[tof < 0] = GlobalConstants.INT_FILLVAL
    vhat_z[tof < 0] = GlobalConstants.INT_FILLVAL

    return vhat_x, vhat_y, vhat_z


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

"""Calculates Extended Raw Events for ULTRA L1b."""

import logging
from collections import defaultdict

import numpy as np
import xarray

from imap_processing.ultra.l1b.lookup_utils import (
    get_back_position,
    get_energy_norm,
    get_image_params,
    get_norm,
    get_y_adjust,
)

logger = logging.getLogger(__name__)


def get_ph_tof_and_back_positions(events_dataset: xarray.Dataset, xf: np.array):
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
    events_dataset : xarray.Dataset
        Data in xarray format.
    xf : np.array
        x front position in (hundredths of a millimeter).

    Returns
    -------
    tof : np.array
        Time of flight (tenths of a nanosecond).
    t2 : float
        Particle time of flight (i.e. from start to stop)
        (tenths of a nanosecond).
    xb : np.array
        Back positions in x direction (hundredths of a millimeter).
    yb : np.array
        Back positions in y direction (hundredths of a millimeter).
    """
    indices = np.where(np.isin(events_dataset["STOP_TYPE"], [1, 2]))[0]

    # There are mismatches between the stop TDCs, i.e., SpN, SpS, SpE, and SpW.
    # This normalizes the TDCs
    sp_n_norm = get_norm(
        events_dataset["STOP_NORTH_TDC"].data[indices], "TpSpNNorm", "ultra45"
    )
    sp_s_norm = get_norm(
        events_dataset["STOP_SOUTH_TDC"].data[indices], "TpSpSNorm", "ultra45"
    )
    sp_e_norm = get_norm(
        events_dataset["STOP_EAST_TDC"].data[indices], "TpSpENorm", "ultra45"
    )
    sp_w_norm = get_norm(
        events_dataset["STOP_WEST_TDC"].data[indices], "TpSpWNorm", "ultra45"
    )

    # Convert normalized TDC values into units of hundredths of a
    # millimeter using lookup tables.
    xb_index = sp_s_norm.values - sp_n_norm.values + 2047
    yb_index = sp_e_norm.values - sp_w_norm.values + 2047

    # Convert xf to a tof offset
    tofx = sp_n_norm.values + sp_s_norm.values
    tofy = sp_e_norm.values + sp_w_norm.values

    # tof is the average of the two tofs measured in the X and Y directions,
    # tofx and tofy
    # Units in tenths of a nanosecond
    t1 = tofx + tofy  # /2 incorporated into scale

    xb = np.zeros(len(events_dataset["STOP_TYPE"]))
    yb = np.zeros(len(events_dataset["STOP_TYPE"]))

    # particle_tof (t2) used later to compute etof
    t2 = np.zeros(len(events_dataset["STOP_TYPE"]))
    tof = np.zeros(len(events_dataset["STOP_TYPE"]))

    # Stop Type: 1=Top, 2=Bottom
    # Convert converts normalized TDC values into units of
    # hundredths of a millimeter using lookup tables.
    index_top = indices[events_dataset["STOP_TYPE"].data[indices] == 1]
    stop_type_top = events_dataset["STOP_TYPE"].data[indices] == 1
    xb[index_top] = get_back_position(xb_index[stop_type_top], "XBkTp", "ultra45")
    yb[index_top] = get_back_position(yb_index[stop_type_top], "YBkTp", "ultra45")

    # Correction for the propagation delay of the start anode and other effects.
    t2[index_top] = get_image_params("TOFSC") * t1[stop_type_top] + get_image_params(
        "TOFTPOFF"
    )
    tof[index_top] = t2[index_top] + xf[stop_type_top] * get_image_params("XFTTOF")

    index_bottom = indices[events_dataset["STOP_TYPE"].data[indices] == 2]
    stop_type_bottom = events_dataset["STOP_TYPE"].data[indices] == 2
    xb[index_bottom] = get_back_position(xb_index[stop_type_bottom], "XBkBt", "ultra45")
    yb[index_bottom] = get_back_position(yb_index[stop_type_bottom], "YBkBt", "ultra45")

    # Correction for the propagation delay of the start anode and other effects.
    t2[index_bottom] = get_image_params("TOFSC") * t1[
        stop_type_bottom
    ] + get_image_params("TOFBTOFF")
    tof[index_bottom] = t2[index_bottom] + xf[stop_type_bottom] * get_image_params(
        "XFTTOF"
    )

    # TODO: why mult by 100?
    tof = tof[indices].astype(np.float64) * 100
    t2 = t2[indices].astype(np.float64)
    xb = xb[indices].astype(np.float64)
    yb = yb[indices].astype(np.float64)

    return tof, t2, xb, yb


def get_front_x_position(start_type: np.array, start_position_tdc: np.array):
    """
    Calculate the front xf position.

    Converts TDC values into units of hundredths of a
    millimeter using a scale factor and offsets.
    Further description is available on pages 30 of
    IMAP-Ultra Flight Software Specification document
    (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    start_type : np.array
        Start Type: 1=Left, 2=Right.
    start_position_tdc: np.array
        Start Position Time to Digital Converter (TDC).

    Returns
    -------
    xf : np.array
        x front position (hundredths of a millimeter).
    """
    indices = np.where((start_type == 1) | (start_type == 2))

    xftsc = get_image_params("XFTSC")
    xft_lt_off = get_image_params("XFTLTOFF")
    xft_rt_off = get_image_params("XFTRTOFF")
    xft_off = np.where(start_type[indices] == 1, xft_lt_off, xft_rt_off)

    # Calculate xf and convert to hundredths of a millimeter
    # Note FSW uses xft_off+1.8, but the lookup table uses xft_off
    # Note FSW uses xft_off-.25, but the lookup table uses xft_off
    xf = (xftsc * -start_position_tdc[indices] + xft_off) * 100

    xf = xf.astype(np.float64)

    return xf


def get_front_y_position(
    events_dataset: xarray.DataArray, yb: np.array
) -> tuple[np.array, np.array]:
    """
    Compute the adjustments for the front y position and distance front to back.

    This function utilizes lookup tables and trigonometry based on
    the angle of the foil. Further description is available in the
    IMAP-Ultra Flight Software Specification document.

    Parameters
    ----------
    events_dataset : xarray.DataArray
        Data in xarray format.
    yb : np.array
        y back position in hundredths of a millimeter.

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
    start_type_left = events_dataset["START_TYPE"].data == 1
    start_type_right = events_dataset["START_TYPE"].data == 2
    index_array = np.arange(len(events_dataset["START_TYPE"]))
    index_left = index_array[start_type_left]
    index_right = index_array[start_type_right]

    yf = np.zeros(len(events_dataset["START_TYPE"]))
    d = np.zeros(len(events_dataset["START_TYPE"]))

    yf_estimate_left = 40.0  # front position of particle for left shutter (mm)
    yf_estimate_right = -40.0  # front position of particle for right shutter (mm)

    # Compute adjustments for left start type
    dy_lut_left = np.round((yf_estimate_left - yb[start_type_left] / 100) * 256 / 81.92)
    y_adjust_left = get_y_adjust(dy_lut_left) / 100  # y adjustment in mm
    yf[index_left] = (
        yf_estimate_left - y_adjust_left
    ) * 100  # hundredths of a millimeter
    distance_adjust_left = (
        np.sqrt(2) * d_slit_foil - y_adjust_left
    )  # distance adjustment in mm
    d[index_left] = (slit_z - distance_adjust_left) * 100  # hundredths of a millimeter

    # Compute adjustments for right start type
    dy_lut_right = np.round(
        (yb[start_type_right] / 100 - yf_estimate_right) * 256 / 81.92
    )
    y_adjust_right = get_y_adjust(dy_lut_right) / 100  # y adjustment in mm
    yf[index_right] = (
        yf_estimate_right + y_adjust_right
    ) * 100  # hundredths of a millimeter
    distance_adjust_right = (
        np.sqrt(2) * d_slit_foil - y_adjust_right
    )  # distance adjustment in mm
    d[index_right] = (
        slit_z - distance_adjust_right
    ) * 100  # hundredths of a millimeter

    d = d.astype(np.float64)
    yf = yf.astype(np.float64)

    return d, yf


def get_coincidence_positions(
    index: int, events_dataset: xarray.Dataset, particle_tof: float
):
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
    index : int
        Index of the event.
    events_dataset : xarray.Dataset
        Data in xarray format.
    particle_tof : float
        Particle time of flight (i.e. from start to stop)
        (tenths of a nanosecond).

    Returns
    -------
    etof : float
        Time for the electrons to travel back to
        coincidence anode (tenths of a nanosecond).
    xc : float
        x coincidence position (hundredths of a millimeter).
    """
    # TODO: This works for the top and bottom anodes (e.g. TpSpNNorm, BtSpNNorm)?

    if (
        events_dataset["COIN_TYPE"].data[index] == 1
        or events_dataset["COIN_TYPE"].data[index] == 2
    ):
        # Normalized TDCs
        # For the stop anode, there are mismatches between the coincidence TDCs,
        # i.e., CoinN and CoinS. They must be normalized via lookup tables.
        coin_n_norm = get_norm(
            events_dataset["COIN_NORTH_TDC"], "TpCoinNNorm", "ultra45"
        )
        coin_s_norm = get_norm(
            events_dataset["COIN_SOUTH_TDC"], "TpCoinSNorm", "ultra45"
        )

    else:
        raise ValueError("Error: Invalid Coincidence Type")

    t1 = coin_n_norm + coin_s_norm  # /2 incorporated into scale

    if events_dataset["COIN_TYPE"].data[index] == 1:
        # calculate x coincidence position xc
        xc = get_image_params("XCoinTpSc") * (
            coin_s_norm - coin_n_norm
        ) / 1024 + get_image_params("XCoinTpOff")  # hundredths of a millimeter
        t2 = get_image_params("eTOFSc") * t1 / 1024 + get_image_params("eTOFTpOff")
    elif events_dataset["COIN_TYPE"].data[index] == 2:
        # calculate x coincidence position xc
        xc = get_image_params("XCoinBtSc") * (
            coin_s_norm - coin_n_norm
        ) / 1024 + get_image_params("XCoinBtOff")  # hundredths of a millimeter
        t2 = get_image_params("eTOFSc") * t1 / 1024 + get_image_params("eTOFBtOff")
    else:
        raise ValueError("Error: Invalid Coin Type")

    # Time for the electrons to travel back to coincidence anode.
    etof = t2 - particle_tof

    return etof, xc


def get_ssd_offset_and_positions(events_dataset: xarray.Dataset):
    """Figure out what SSD a particle hit.

    Parameters
    ----------
    events_dataset : xarray.Dataset
        Data in xarray format.

    Returns
    -------
    ssd_indices : np.array
        Index of SSD
    yb : np.array
        y ssd position (hundredths of a millimeter).
    tof_offsets : np.array
        Time of flight offset (nanoseconds).
    """
    ssd_indices = np.array([], dtype=int)
    ybs = np.array([], dtype=np.float64)
    tof_offsets = np.array([], dtype=np.float64)
    ssds = np.array([], dtype=int)

    # START_TYPE: 1=Left
    indices = np.where(
        (events_dataset["STOP_TYPE"] >= 8) & (events_dataset["START_TYPE"] == 1)
    )[0]
    for i in range(8):
        ssd_index = indices[events_dataset[f"SSD_FLAG_{i}"].data[indices] == 1]
        ssd_indices = np.concatenate((ssd_indices, ssd_index))

        ssds = np.concatenate((ssds, np.full(len(ssd_index), i, dtype=int)))

        yb = np.full(len(ssd_index), get_image_params(f"YBKSSD{i}"))
        ybs = np.concatenate((ybs, yb))

        tof_offset = np.full(len(ssd_index), get_image_params(f"TOFSSDLTOFF{i}"))
        tof_offsets = np.concatenate((tof_offsets, tof_offset))

    # START_TYPE: 2=Right
    indices = np.where(
        (events_dataset["STOP_TYPE"] >= 8) & (events_dataset["START_TYPE"] == 2)
    )[0]
    for i in range(8):
        ssd_index = indices[events_dataset[f"SSD_FLAG_{i}"].data[indices] == 1]
        ssd_indices = np.concatenate((ssd_indices, ssd_index))

        ssds = np.concatenate((ssds, np.full(len(ssd_index), i, dtype=int)))

        yb = np.full(len(ssd_index), get_image_params(f"YBKSSD{i}"))
        ybs = np.concatenate((ybs, yb))

        tof_offset = np.full(len(ssd_index), get_image_params(f"TOFSSDRTOFF{i}"))
        tof_offsets = np.concatenate((tof_offsets, tof_offset))

    # multiply ybs times 100 to convert to hundredths of a millimeter.
    return ssd_indices, ybs * 100, tof_offsets, ssds


def get_ssd_tof(events_dataset: xarray.Dataset, xf: np.array):
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
    events_dataset : xarray.Dataset
        Data in xarray format.
    xf : np.array
        Front x position (hundredths of a millimeter)

    Returns
    -------
    tof : int
        Time of flight (tenths of a nanosecond).
    """
    ssd_indices, ybs, tof_offsets, ssd = get_ssd_offset_and_positions(events_dataset)

    # in nanoseconds
    time = (
        get_image_params("TOFSSDSC")
        * events_dataset["COIN_DISCRETE_TDC"].data[ssd_indices]
        + tof_offsets
    )

    # The scale factor and offsets, and a multiplier to convert xf to a tof offset.
    tof = (
        time
        + get_image_params("TOFSSDTOTOFF")
        + xf[ssd_indices] * get_image_params("XFTTOF")
    )

    tof = tof.astype(np.float64)

    return ssd_indices, tof, ssd


def get_energy_pulse_height(events_dataset: xarray.Dataset, xb: np.array, yb: np.array):
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
    events_dataset : xarray.Dataset
        Data in xarray format.
    xb : np.array
        x back position (hundredths of a millimeter).
    yb : np.array
        y back position (hundredths of a millimeter).

    Returns
    -------
    energy : np.array
        Energy measured using the pulse height
        from the stop anode (DN).
    """
    indices_1 = np.where(events_dataset["STOP_TYPE"] == 1)[0]
    indices_2 = np.where(events_dataset["STOP_TYPE"] == 2)[0]

    # Stop type 1
    xlut = (xb[indices_1] / 100 - 25 / 2) * 20 / 50  # mm
    ylut = (yb[indices_1] / 100 + 82 / 2) * 32 / 82  # mm
    energy_1 = xlut + ylut  # placeholder
    # energy = events_dataset["ENERGY_PH"].data[indices_1] -
    # get_image_params("SpTpPHOffset")
    # TODO * SpTpPHCorr[xlut, ylut] / 1024; George Clark working on it

    # Stop type 2
    xlut = (xb[indices_2] / 100 + 50 + 25 / 2) * 20 / 50  # mm
    ylut = (yb[indices_2] / 100 + 82 / 2) * 32 / 82  # mm
    energy_2 = xlut + ylut  # placeholder
    # energy = events_dataset["ENERGY_PH"].data[indices_2] -
    # energy = pulse_height - get_image_params("SpBtPHOffset")
    # TODO * SpBtPHCorr[xlut, ylut]/1024; George Clark working on it

    energy = np.concatenate((energy_1, energy_2))
    energy = energy.astype(np.float64)

    return energy


def get_energy_ssd(
    events_dataset: xarray.Dataset, ssd_indices: np.array, ssd: np.array
):
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
    events_dataset: xarray.Dataset
        Events dataset.
    ssd_indices : np.array
        Indices of the event.
    ssd : np.array
        SSD number.

    Returns
    -------
    energy : float
        Energy measured using the SSD.
    """
    # TODO: find a reference for this
    composite_energy_threshold = 1706

    energy = events_dataset["ENERGY_PH"].data[ssd_indices]

    composite_energy = np.empty(len(energy), dtype=np.float64)

    composite_energy[energy >= composite_energy_threshold] = (
        composite_energy_threshold
        + events_dataset["PULSE_WIDTH"][ssd_indices][
            energy >= composite_energy_threshold
        ]
    )
    composite_energy[energy < composite_energy_threshold] = energy[
        energy < composite_energy_threshold
    ]

    energy_norm = get_energy_norm(ssd, composite_energy)

    return energy_norm


def determine_species_ssd(energy: np.array, tof: np.array, r: np.array):
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
        Time of flight of the SSD event (tenths of a nanosecond)
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

    bin = 0  # placeholder

    # TODO: get these lookup tables
    # if r < get_image_params("PathSteepThresh"):
    #     # bin = ExTOFSpeciesSteep[energy, ctof]
    # elif r < get_image_params("PathMediumThresh"):
    #     # bin = ExTOFSpeciesMedium[energy, ctof]
    # else:
    #     # bin = ExTOFSpeciesFlat[energy, ctof]

    # Convert ctof from nanoseconds to tenths of a nanosecond
    return ctof * 10, bin


def determine_species_pulse_height(energy: np.array, tof: np.array, r: np.array):
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
    bin = 0  # placeholder
    # bin = PHxTOFSpecies[ctof, energy]

    return ctof, bin


def get_particle_velocity(
    front_position: tuple, back_position: tuple, d: np.array, tof: np.array
):
    """
    Determine the particle velocity.

    The equation is: velocity = ((xf - xb), (yf - yb), d).

    Further description is available on pages 39 of
    IMAP-Ultra Flight Software Specification document
    (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    front_position : tuple of floats
        Front position (xf,yf) (hundredths of a millimeter).
    back_position : tuple of floats
        Back position (xb,yb) (hundredths of a millimeter).
    d : np.array
        distance from slit to foil (hundredths of a millimeter).
    tof : np.array
        Time of flight (tenths of a nanosecond).

    Returns
    -------
    vhat_x : np.array.
        Normalized component of the velocity vector in x direction.
    vhat_y : np.array.
        Normalized component of the velocity vector in y direction.
    vhat_z : np.array.
        Normalized component of the velocity vector in z direction.
    """
    # FSW seems to make this absolute value. I will do the same for now.
    # TODO: Ask Ultra team about this.
    if tof[tof < 0].any():
        tof = abs(tof)
        logging.info("Negative tof values found.")

    delta_x = front_position[0] - back_position[0]
    delta_y = front_position[1] - back_position[1]

    v_x = delta_x / tof
    v_y = delta_y / tof
    v_z = d / tof

    # Magnitude of the velocity vector
    magnitude_v = np.sqrt(v_x**2 + v_y**2 + v_z**2)

    vhat_x = v_x / magnitude_v
    vhat_y = v_y / magnitude_v
    vhat_z = v_z / magnitude_v

    return vhat_x, vhat_y, vhat_z


def get_path_length(front_position, back_position, d):
    """Calculate the path length.

    Parameters
    ----------
    front_position : tuple of floats
        Front position (xf,yf) (hundredths of a millimeter).
    back_position : tuple of floats
        Back position (xb,yb) (hundredths of a millimeter).
    d : float
        distance from slit to foil (hundredths of a millimeter).

    Returns
    -------
    r : float
        Path length (hundredths of a millimeter).
    """
    r = np.sqrt(
        (front_position[0] - back_position[0]) ** 2
        + (front_position[1] - back_position[1]) ** 2
        + (d) ** 2
    )

    return r


def get_extended_raw_events(events_dataset):
    """
    Create dictionary of extended raw events.

    Parameters
    ----------
    events_dataset : dict
        Data in xarray format.

    Returns
    -------
    data_dict : dict of lists
        Data for extended raw events.

    Definitions:
    START_TYPE = 1: Left Slit
    START_TYPE = 2: Right Slit
    STOP_TYPE = 1: Top
    STOP_TYPE = 2: Bottom
    COIN_TYPE = 1: Top
    COIN_TYPE = 2: Bottom
    STOP_TYPE >= 8: SSD

    TODO: stop type 1, 2, 8-15; nothing else?
    TODO: how should event type be formatting? See ppt
    presentation Extended Raw Events
    Table for more details
    """
    data_dict = defaultdict(list)

    for time in events_dataset["SHCOARSE"].data:
        index = np.where(events_dataset["SHCOARSE"].data == time)[0][0]
        count = events_dataset["COUNT"].data[index]

        if count == 0:
            process_count_zero(data_dict)
            continue  # TODO: handle as needed: -1?

        # Shared processing for valid start types
        start_type = events_dataset["START_TYPE"].data[index]
        if start_type not in [1, 2]:
            raise ValueError("Error: Invalid Start Type")

        start_position_tdc = events_dataset["START_POS_TDC"].data[index]
        xf = get_front_x_position(start_type, start_position_tdc)
        stop_type = events_dataset["STOP_TYPE"].data[index]

        if stop_type in [1, 2]:
            # Process for Top and Bottom stop types
            tof, particle_tof, xb, yb = get_ph_tof_and_back_positions(
                index, events_dataset, xf
            )
            d, yf = get_front_y_position(start_type, yb)
            pulse_height = events_dataset["ENERGY_PH"].data[index]
            # TODO stopped here
            energy = get_energy_pulse_height(pulse_height, stop_type, xb, yb)
            r = get_path_length(xf, xb, yf, yb, d)
            ctof, bin = determine_species_pulse_height(energy, tof, r)
            velocity = get_particle_velocity(xf, xb, yf, yb, d)
        elif stop_type >= 8:
            # Process for SSD stop types
            xb, yb, tof = get_ssd_tof(index, events_dataset, xf)
            d, yf = get_front_y_position(start_type, yb)
            energy = get_energy_ssd(index, events_dataset)
            r = get_path_length(xf, xb, yf, yb, d)
            ctof, bin = determine_species_ssd(energy, tof, r)
            velocity = get_particle_velocity(xf, xb, yf, yb, d)
        else:
            raise ValueError("Error: Invalid Stop Type")

        # Append to dictionary
        data_dict["front_position"].append((xf, yf))
        data_dict["back_position"].append((xb, yb))

        data_dict["tof"].append(tof)
        data_dict["energy"].append(energy)

        # Determine_species independenty of event data
        data_dict["species"].append(bin)
        data_dict["velocity"].append(velocity)

        coincidence_type = events_dataset["COIN_TYPE"].data[index]
        if coincidence_type in [1, 2]:
            etof, xc = get_coincidence_positions(index, events_dataset, particle_tof)

            # Append to dictionary
            data_dict["coincidence_position"].append((xc, yc))
            data_dict["etof"].append(etof)
        else:
            data_dict["coincidence_position"].append((-1, -1))
            data_dict["etof"].append(-1)
            logger.info("Coincidence position not equal to top or bottom.")

    return data_dict

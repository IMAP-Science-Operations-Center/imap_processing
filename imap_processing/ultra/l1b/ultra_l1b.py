"""Calculates Extended Raw Events for ULTRA L1b."""

# TODO: Decide on consistent fill values.
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


def get_back_positions(index: int, events_dataset: xarray.Dataset, xf: float):
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
    index : int
        Index of the event.
    events_dataset : xarray.Dataset
        Data in xarray format.
    xf : float
        x front position in (hundredths of a millimeter).

    Returns
    -------
    tof : float
        Time of flight (tenths of a nanosecond).
    particle_tof : float
        Particle time of flight (i.e. from start to stop)
        (tenths of a nanosecond).
    xb : float
        x back position in (hundredths of a millimeter).
    yb : float
        y back position in (hundredths of a millimeter).
    """
    # There are mismatches between the stop TDCs, i.e., SpN, SpS, SpE, and SpW.
    # This normalizes the TDCs
    sp_n_norm = get_norm(
        events_dataset["STOP_NORTH_TDC"].data[index], "TpSpNNorm", "ultra45"
    )
    sp_s_norm = get_norm(
        events_dataset["STOP_SOUTH_TDC"].data[index], "TpSpSNorm", "ultra45"
    )
    sp_e_norm = get_norm(
        events_dataset["STOP_EAST_TDC"].data[index], "TpSpENorm", "ultra45"
    )
    sp_w_norm = get_norm(
        events_dataset["STOP_WEST_TDC"].data[index], "TpSpWNorm", "ultra45"
    )

    # Convert normalized TDC values into units of hundredths of a
    # millimeter using lookup tables.
    xb_index = sp_s_norm - sp_n_norm + 2047
    yb_index = sp_e_norm - sp_w_norm + 2047

    # Convert xf to a tof offset,
    tofx = sp_n_norm + sp_s_norm
    tofy = sp_e_norm + sp_w_norm

    # tof is the average of the two tofs measured in the X and Y directions,
    # tofx and tofy
    # Units in tenths of a nanosecond
    t1 = tofx + tofy  # /2 incorporated into scale

    # Stop Type: 1=Top, 2=Bottom
    if events_dataset["STOP_TYPE"].data[index] == 1:
        # TODO: Ask if this if statement is needed.
        # If the difference between TOFy and TOFx is too large, the event is rejected
        if (
            get_image_params("TOFDiffTpMin")
            <= (tofy - tofx)
            <= get_image_params("TOFDiffTpMax")
        ):
            # Convert converts normalized TDC values into units of
            # hundredths of a millimeter using lookup tables.
            xb = get_back_position(xb_index, "XBkTp", "ultra45")
            yb = get_back_position(yb_index, "YBkTp", "ultra45")

            # Correction for the propagation delay of the start anode and other effects.
            t2 = get_image_params("TOFSc") * t1 / 1024 + get_image_params("TOFTpOff")
        else:
            # TODO: add xb and yb here?
            xb = yb = -1
            logger.info("Event Rejected due to TOFDiffBt Min/Max.")

    elif events_dataset["STOP_TYPE"].data[index] == 2:
        # If the difference between TOFy and TOFx is too large, the event is rejected
        if (
            get_image_params("TOFDiffBtMin")
            <= (tofy - tofx)
            <= get_image_params("TOFDiffBtMax")
        ):
            xb = get_back_position(xb_index, "XBkBt", "ultra45")
            yb = get_back_position(yb_index, "YBkBt", "ultra45")

            # Correction for the propagation delay of the start anode and other effects.
            t2 = get_image_params("TOFSc") * t1 / 1024 + get_image_params("TOFBtOff")
        else:
            # TODO: add xb=-1 and yb=-1 here?
            xb = yb = -1
            logger.info("Event Rejected due to TOFDiffBt Min/Max.")
    else:
        raise ValueError("Error: Invalid Stop Type")

    # Correction for the propagation delay of the start anode and other effects (t2)
    tof = t2 + xf * get_image_params("XFtTOF") / 32768
    particle_tof = t2  # used later to compute etof

    return tof, particle_tof, xb, yb


def get_front_x_position(start_type: int, start_position_tdc: float):
    """
    Calculate the front xf position.

    Converts TDC values into units of hundredths of a
    millimeter using a scale factor and offsets.
    Further description is available on pages 30 of
    IMAP-Ultra Flight Software Specification document
    (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    start_type : int
        Start Type: 1=Left, 2=Right.
    start_position_tdc: float
        Start Position Time to Digital Converter (TDC).

    Returns
    -------
    xf : float
        x front position (hundredths of a millimeter).
    """
    # A particle entering the left shutter will trigger
    # the left anode and vice versa.
    if start_type == 1:
        xf = get_image_params("XFtSc") * -start_position_tdc / 1024 + get_image_params(
            "XFtLtOff"
        )
    elif start_type == 2:
        xf = get_image_params("XFtSc") * -start_position_tdc / 1024 + get_image_params(
            "XFtRtOff"
        )
    else:
        raise ValueError("Error: Invalid Start Type")

    return xf


def get_front_y_position(start_type: int, yb: float):
    """
    Compute the adjustments.

    These are needed for the front position
    (yf) and distance front to back (d).
    This utilizes lookup tables, but is based on
    the angle of the foil and trigonometry.
    Further description is available on pages 38-39 of
    IMAP-Ultra Flight Software Specification document
    (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    start_type : int
        Start Type: 1=Left, 2=Right.
    yb : float
        y back position in (hundredths of a millimeter).

    Returns
    -------
    d : float
        Distance front to back (mm).
    yf : float
        Front y position (mm).
    """
    df = 3.39  # shortest distance from slit to foil (mm)
    ds = 45  # shutters slit to back detectors (mm)
    yf_estimate = 40  # from position of particle (mm)

    # A particle entering the left shutter will trigger
    # the left anode and vice versa.
    if start_type == 1:
        # TODO: make certain yb units correct
        dy_lut = (yf_estimate - yb / 100) * 256 / 81.92  # mm
        yadj = get_y_adjust(dy_lut)  # mm
        yf = yf_estimate - yadj  # mm
    elif start_type == 2:
        dy_lut = (yb / 100 - yf_estimate) * 256 / 81.92  # mm
        yadj = get_y_adjust(dy_lut)  # mm
        yf = yf_estimate + yadj  # mm
    else:
        raise ValueError("Error: Invalid Start Type")

    dadj = np.sqrt(2) * df - yadj  # mm
    d = ds - dadj  # mm

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
    TODO: where is yc defined in document?
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


def get_ssd_index(index: int, events_dataset: xarray.Dataset):
    """Figure out what SSD a particle hit.

    Parameters
    ----------
    index : int
        Index of the event.
    events_dataset : xarray.Dataset
        Data in xarray format.

    Returns
    -------
    ssd_index : int
        Index of SSD
    """
    ssd_indices = [
        i for i in range(8) if events_dataset[f"SSD_FLAG_{i}"].data[index] == 1
    ]

    if len(ssd_indices) == 1:
        ssd_flag_index = ssd_indices[0]
        logger.info(
            f"Exactly one of the values is equal to 1, "
            f"found at SSD_FLAG_{ssd_flag_index}."
        )
    elif not ssd_indices:
        raise ValueError("No SSD Flag found equal to 1.")
    else:
        raise ValueError("More than one SSD Flag found equal to 1.")

    ssd_index = ssd_indices[0]

    return ssd_index


def get_ssd_positions(index: int, events_dataset: xarray.Dataset, xf: float):
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
    index : int
        Index of the event.
    events_dataset : xarray.Dataset
        Data in xarray format.
    xf : int
        Front x position (hundredths of a millimeter)

    Returns
    -------
    xb : int
        x back position (hundredths of a millimeter).
    yb : int
        y back position (hundredths of a millimeter).
    tof : int
        Time of flight (tenths of a nanosecond).
    """
    xb = 0

    # Start Type: 1=Left, 2=Right
    if events_dataset["START_TYPE"].data[index] == 1:  # Left
        side = "Lt"
    elif events_dataset["START_TYPE"].data[index] == 2:  # Right
        side = "Rt"

    coin_d = get_image_params("COIN_DISCRETE_TDC")

    ssd_index = get_ssd_index(index, events_dataset)

    yb = get_image_params(f"Y Back SSD Position {ssd_index[0]}")
    tof_offset = get_image_params(f"TOF SSD {side} Offset {ssd_index[0]}")
    time = get_image_params("TOFSSDSc") * coin_d / 1024 + tof_offset
    # The scale factor and offsets, and a multiplier to convert xf to a tof offset.
    # TODO: make certain xf is in expected units
    tof = (
        time
        + get_image_params("TOFSSDTotOff")
        + xf * get_image_params("XFtTOF") / 32768
    )

    return xb, yb, tof


def get_energy_pulse_height(pulse_height: float, stop_type: int, xb: float, yb: float):
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
    pulse_height: float
        Pulse height from the stop anode.
        # TODO: units?
    stop_type: int
        Stop Type: 1=Top, 2=Bottom.
    xb : float
        x back position (hundredths of a millimeter).
    yb : float
        y back position (hundredths of a millimeter).

    Returns
    -------
    energy : float
        Energy measured using the pulse height
        from the stop anode (keV).
    # TODO: make certain units are correct.
    """
    if stop_type == 1:
        # TODO: make certain xb, yb units correct
        xlut = (xb / 100 - 25 / 2) * 20 / 50  # mm
        ylut = (yb / 100 + 82 / 2) * 32 / 82  # mm
        energy = xlut + ylut  # placeholder
        # energy = pulse_height - get_image_params("SpTpPHOffset")
        # TODO * SpTpPHCorr[Xlut, Ylut] / 1024; George Clark working on it
    elif stop_type == 2:
        # TODO: make certain xb, yb units correct
        xlut = (xb / 100 + 50 + 25 / 2) * 20 / 50  # mm
        ylut = (yb / 100 + 82 / 2) * 32 / 82  # mm
        energy = xlut + ylut  # placeholder
        # energy = pulse_height - get_image_params("SpBtPHOffset")
        # TODO * SpBtPHCorr[Xlut, Ylut]/1024; George Clark working on it
    else:
        raise ValueError("Error: Invalid Stop Type")

    return energy


def get_energy_ssd(index: int, events_dataset: xarray.Dataset):
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
    index : int
        Index of the event.
    events_dataset : xarray.Dataset
        Data in xarray format.

    Returns
    -------
    energy : float
        Energy measured using the SSD (keV).
    #TODO: make certain units are correct.
    """
    # TODO: find a reference for this
    composite_energy_threshold = 1706

    energy = events_dataset["EnergyPH"].data[index]

    if energy < composite_energy_threshold:
        composite_energy = energy
    else:
        composite_energy = (
            composite_energy_threshold + events_dataset["PULSE_WIDTH"].data[index]
        )

    ssd_index = get_ssd_index(index, events_dataset)
    energy_norm = get_energy_norm(ssd_index, composite_energy)

    return energy_norm


def determine_species_ssd(energy: float, tof: float, r: float):
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
    energy : float
        Energy from the SSD event (keV).
    tof : float
        Time of flight of the SSD event (tenths of a nanosecond)
    r : float
        Path length (mm).

    Returns
    -------
    ctof : float
        Corrected TOF.
    bin : int
        Species bin.
    """
    ctof = tof  # * dmin-ssd-ctof / r TODO: this is unknown dmin-ssd-ctof

    # TODO: get these lookup tables
    if r < get_image_params("PathSteepThresh"):
        # bin = ExTOFSpeciesSteep[energy, ctof]
        bin = 0  # placeholder
    elif r < get_image_params("PathMediumThresh"):
        # bin = ExTOFSpeciesMedium[energy, ctof]
        bin = 0  # placeholder
    else:
        # bin = ExTOFSpeciesFlat[energy, ctof]
        bin = 0  # placeholder

    return ctof, bin


def determine_species_pulse_height(energy: float, tof: float, r: float):
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
    energy : float
        Energy from the SSD event (keV).
    tof : float
        Time of flight of the SSD event (tenths of a nanosecond).
    r : float
        Path length (mm).

    Returns
    -------
    ctof : float
        Corrected TOF.
    bin : int
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


def get_particle_velocity(xf: float, xb: float, yf: float, yb: float, d: float):
    """
    Determine the particle velocity.

    The equation is: velocity = ((xf - xb), (yf - yb), d).

    Further description is available on pages 39 of
    IMAP-Ultra Flight Software Specification document
    (7523-9009_Rev_-.pdf).

    Parameters
    ----------
    xf : float
        x front position (hundredths of a millimeter).
    xb : float
        x back position (hundredths of a millimeter).
    yf : float
        Front y position (mm).
    yb : float
        y back position (hundredths of a millimeter).
    d : float
        distance from slit to foil (mm).

    Returns
    -------
    velocity : tuple of floats.
        Corrected TOF.
    """
    # TODO: where is the time component?
    velocity = ((xf / 100 - xb / 100), (yf - yb / 100), d)

    return velocity


def process_count_zero(data_dict: dict):
    """
    Append default values for events with count == 0.

    Parameters
    ----------
    data_dict : dict
        Data in dictionary format.
    """
    data_dict["front_position"].append((-1, -1))
    data_dict["back_position"].append((-1, -1))
    data_dict["tof"].append(-1)
    data_dict["energy"].append(-1)
    data_dict["species"].append(-1)
    data_dict["velocity"].append((-1, -1, -1))


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
            tof, particle_tof, xb, yb = get_back_positions(index, events_dataset, xf)
            d, yf = get_front_y_position(index, events_dataset, yb)
            pulse_height = events_dataset["ENERGY_PH"].data[index]
            energy = get_energy_pulse_height(pulse_height, stop_type, xb, yb)
            r = np.sqrt((xf / 100 - xb / 100) ** 2 + (yf - yb / 100) ** 2 + d**2)
            ctof, bin = determine_species_pulse_height(energy, tof, r)
            velocity = get_particle_velocity(xf, xb, yf, yb, d)
        elif stop_type >= 8:
            # Process for SSD stop types
            xb, yb, tof = get_ssd_positions(index, events_dataset, xf)
            d, yf = get_front_y_position(index, events_dataset, yb)
            energy = get_energy_ssd(index, events_dataset)
            r = np.sqrt((xf / 100 - xb / 100) ** 2 + (yf - yb / 100) ** 2 + d**2)
            ctof, bin = determine_species_ssd(energy, tof, r)
            velocity = get_particle_velocity(xf, xb, yf, yb, d)
        else:
            raise ValueError("Error: Invalid Stop Type")

        # Append to dictionary
        data_dict["front_position"].append((xf, yf))
        data_dict["back_position"].append((xb, yb))
        # TODO: is this the tof we want to keep or ctof?
        data_dict["tof"].append(tof)
        data_dict["energy"].append(energy)
        # TODO: should we take this from determine_species or event data?
        data_dict["species"].append(bin)
        data_dict["velocity"].append(velocity)

        coincidence_type = events_dataset["COIN_TYPE"].data[index]
        if coincidence_type in [1, 2]:
            etof, xc = get_coincidence_positions(index, events_dataset, particle_tof)
            # TODO: equation for yc
            yc = 0  # placeholder

            # Append to dictionary
            data_dict["coincidence_position"].append((xc, yc))
            data_dict["etof"].append(etof)
        else:
            data_dict["coincidence_position"].append((-1, -1))
            data_dict["etof"].append(-1)
            logger.info("Coincidence position not equal to top or bottom.")

    return data_dict

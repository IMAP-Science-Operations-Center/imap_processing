"""Naming utilities for ENA maps."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Literal

from imap_processing.spice.geometry import SpiceFrame


class MappableInstrumentShortName(Enum):
    """Enumeration of the short names of the ENA and other mappable instruments."""

    # Lo has several different modes
    LO = "ilo"
    LO_HI_RES = "l"
    LO_HI_THROUGHPUT = "t"

    HI = "h"
    ULTRA = "u"
    IDEX = "idx"
    GLOWS = "glx"


# Ignore too many branches warning
def build_l2_map_descriptor(  # noqa: PLR0912
    *,
    # Basic parameters must always be passed in as keyword arguments
    instrument: MappableInstrumentShortName,  # Updated to use the enum
    frame: Literal["sf", "hf", "hk"] | SpiceFrame,
    resolution_str: str,
    duration: str | int | timedelta,
    # The rest of the parameters have default values corresponding to the
    # most general cases
    sensor: int | Literal["45", "90", "combined", "ic", "lc", ""] | str = "",
    principal_data: Literal["ena", "spx", "isn", "int", "drt"] = "ena",
    species: Literal["h", "he", "o", "uv", "dust"] = "h",
    survival_corrected: Literal["nsp", "sp"] = "nsp",
    spin_phase: Literal["full", "ram", "anti"] = "full",
    coordinate_system: Literal["hae", "hgi", "rc"] = "hae",
) -> str:
    """
    Build a map descriptor string for the L2 ENA maps.

    Example descriptor string and its meaning:
    "h45-ena-h-hf-sp-ram-hae-6deg-3mo" is:
    An IMAP-HI map made from data taken by HI's 45 degree sensor, of hydrogen ENAs,
    in the heliospheric frame, survival probability corrected,
    of counts in the ram direction, in heliocentric aries ecliptic coordinates,
    with a pixel spacing of 6 degrees, with a duration of 3 months.

    This function requires a large amount of information to be passed in, and will
    likely be used indirectly by methods of a SkyMap object or similar.
    On its own, it defines the standard for the map descriptor string.

    Parameters
    ----------
    instrument : MappableInstrumentShortName
        The instrument name. Instrument and sensor are combined to form the
        instrument descriptor (e.g. "u90", "h45", "idx").
    frame : Literal["sf", "hf", "hk"] | SpiceFrame
        The frame of reference for the map.
        sf: Spacecraft frame.
        hf: Heliospheric frame.
        hk: Heliospheric kinematic frame.
    resolution_str : str
        The resolution of the map as a string.
        For Healpix maps, this is the nside value as "nside128", "nside32", etc.
        For rectangular maps, this is the spacing in degrees as "2deg", "6deg", etc.
    duration : str | int | timedelta
        The duration of the map as a string, and integer number of days, or a timedelta.
        The string should be in the format of "1yr", "6mo", "3mo", etc.
    sensor : int | Literal["45", "90", "combined", "ic", "lc", ""] | str
        The sensor number for the map. By default, this is "".
        For LO, this should be a 3 character string or an integer
        which will be converted to a 3 character string (90 --> "090").
        For Hi and Ultra, this should be "45", "90", or "combined".
        If "combined" is passed, the Hi sensor will be "ic"
        and the Ultra sensor will be "lc".
    principal_data : Literal["ena", "spx", "isn", "int", "drt"]
        The principal data type for the map. By default, this is "ena".
        ena: ENA Intensity.
        spx: Spectral Index.
        isn: ISN Count Rate.
        int: GLOWS Intensity.
        drt: IDEX Dust Rate.
    species : Literal["h", "he", "o", "uv", "dust"]
        The species for the map. By default, this is "h".
        h: Hydrogen.
        he: Helium.
        o: Oxygen.
        uv: UV.
        dust: Dust.
    survival_corrected : Literal["nsp", "sp"]
        Whether the map is survival probability corrected ("sp") or not ("nsp").
    spin_phase : Literal["full", "ram", "anti"]
        The spin phase for the map. By default, this is "full".
        full: Full spin.
        ram: Ram direction only.
        anti: Anti-ram direction only.
    coordinate_system : Literal["hae", "hgi", "rc"]
        The coordinate system for the map. By default, this is "hae".
        hae: Heliocentric Aries Ecliptic.
        Other options are hgi and rc.

    Returns
    -------
    str
        The map descriptor string.
    """
    # Handle sensor
    # Lo may pass in an integer, which should be converted to a 3 character string
    if isinstance(sensor, int):
        if instrument in [
            MappableInstrumentShortName.LO,
            MappableInstrumentShortName.LO_HI_RES,
            MappableInstrumentShortName.LO_HI_THROUGHPUT,
        ]:
            sensor = f"{sensor:03}"
        else:
            raise ValueError("Integer sensor values are only valid for LO instruments.")
    # Hi and Ultra may be either "45", "90", or "combined", in which case
    # Hi should get the sensor "ic" and Ultra should get the sensor "lc"
    elif sensor == "combined":
        if instrument is MappableInstrumentShortName.ULTRA:
            sensor = "lc"
        elif instrument is MappableInstrumentShortName.HI:
            sensor = "ic"

    # Get the instrument descriptor (e.g. "u90", "h45", "ilo")
    instrument_descriptor = f"{instrument.value}{sensor}"

    # Handle duration
    if isinstance(duration, timedelta):
        # Convert timedelta to a string representation of number of 28.5 day months
        num_months = int(duration.days // 28.5)
        duration = f"{num_months}mo"
    elif isinstance(duration, int):
        # Assume number of days and convert to 28.5-day months
        duration = f"{int(duration // 28.5)}mo"
    elif isinstance(duration, str):
        pass
    # Replace 12mo with 1yr
    if duration == "12mo":
        duration = "1yr"

    # Handle frame if passed in as a spice frame
    if isinstance(frame, SpiceFrame):
        match frame:
            case SpiceFrame.IMAP_DPS:
                frame_descriptor = "sf"
            case SpiceFrame.ECLIPJ2000:
                frame_descriptor = "hf"

            # TODO: Handle other frames as required
            case _:
                raise NotImplementedError(f"Frame {frame} is not yet implemented.")
    else:
        frame_descriptor = frame

    map_descriptor = (
        f"{instrument_descriptor}-{principal_data}-{species}-{frame_descriptor}"
        f"-{survival_corrected}-{spin_phase}-{coordinate_system}"
        f"-{resolution_str}-{duration}"
    )
    return map_descriptor


def build_friendly_date_descriptor(
    start_datestring: str,
    duration_months: int,
) -> str:
    """
    Build a friendly date descriptor string for the L2 ENA maps.

    Example descriptor string and its meaning:
    "202607m06" is:
    a map made from data starting in July 2026 with a duration of 6 months.

    Parameters
    ----------
    start_datestring : str
        The start date of the map in one of the following formats:
        YYYY-MM-DDTHH:MM:SS, YYYY-MM-DD, YYYY-MM.
    duration_months : int
        The duration of the map in months.
        This is expected (but not enforced) to be a
        positive integer <= 99.

    Returns
    -------
    str
        The friendly date descriptor string.
    """
    # Parse the start date string with or without days and time

    accepted_formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%Y-%m",
    ]
    for format in accepted_formats:
        try:
            start_date = datetime.strptime(start_datestring, format)
            break
        except ValueError:
            continue

    return f"{start_date.strftime('%Y%m')}m{duration_months:02}"

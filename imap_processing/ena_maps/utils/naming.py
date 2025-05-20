"""Naming utilities for ENA maps."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Literal, cast

from imap_processing.ena_maps import ena_maps
from imap_processing.spice.geometry import SpiceFrame

# Set a constant number of days in a month to calculate the duration of maps
DAYS_IN_MONTH = 30


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


_sensor_types = int | Literal["45", "90", "combined", "ic", "lc", ""]


def get_instrument_descriptor(
    instrument: MappableInstrumentShortName,
    sensor: _sensor_types = "",
) -> str:
    """
    Get the instrument descriptor string for a given instrument and sensor (e.g. "u45").

    Parameters
    ----------
    instrument : MappableInstrumentShortName
        The short name Enum of the instrument.
    sensor : _sensor_types, optional
        The sensor identifier, by default "".

    Returns
    -------
    str
        The instrument descriptor string.
        E.g. "u90", "u45", "ulc", "h45", "ilo", "t075".

    Raises
    ------
    ValueError
        If the sensor is an integer and the instrument is not LO.
    """
    # Handle sensor
    # Lo may pass in an integer, which should be converted to a 3 character string
    if isinstance(sensor, int):
        if instrument in [
            MappableInstrumentShortName.LO,
            MappableInstrumentShortName.LO_HI_RES,
            MappableInstrumentShortName.LO_HI_THROUGHPUT,
        ]:
            sensor_string = f"{sensor:03}"
        else:
            raise ValueError("Integer sensor values are only valid for LO instruments.")
    # Hi and Ultra may be either "45", "90", or "combined", in which case
    # Hi should get the sensor "ic" and Ultra should get the sensor "lc"
    elif sensor == "combined":
        if instrument is MappableInstrumentShortName.ULTRA:
            sensor_string = "lc"
        elif instrument is MappableInstrumentShortName.HI:
            sensor_string = "ic"
    else:
        sensor_string = str(sensor)

    # Get the instrument descriptor (e.g. "u90", "h45", "ilo")
    instrument_descriptor = f"{instrument.value}{sensor_string}"
    return instrument_descriptor


def parse_instrument_descriptor(
    instrument_descriptor: str,
) -> tuple[MappableInstrumentShortName, _sensor_types]:
    """
    Parse the instrument descriptor string into instrument, sensor str reprs.

    Parameters
    ----------
    instrument_descriptor : str
        The instrument descriptor string to parse.

    Returns
    -------
    tuple[str, _sensor_types]
        A tuple containing the instrument short name and the sensor.

    Raises
    ------
    ValueError
        If the instrument descriptor format can't be parsed as a combined instrument or
        as a regex match to the expected format.
    """
    # Default to no sensor
    sensor: _sensor_types = ""

    if instrument_descriptor.endswith("c"):
        sensor = "combined"
        instrument_short_name = instrument_descriptor[:-2]
        instrument = MappableInstrumentShortName(instrument_short_name)
    else:
        # Do regex to get the instrument and sensor:
        # The first 1 or 3 characters are the instrument short name
        # if the last 2 or 3 characters are digits, then this is a sensor
        # otherwise, the sensor is empty str
        match = re.match(r"([a-z]{1,3})(\d{2,3})?", instrument_descriptor)
        if match:
            instrument_short_name = match.group(1)
            sensor_match = str(match.group(2)) if match.group(2) else ""

            # If sensor is 2 digits, it must be either 45 or 90
            if sensor_match and len(sensor_match) == 2:
                sensor = cast(_sensor_types, sensor_match)
            # If sensor is 3 digits, convert to int
            elif sensor_match and len(sensor_match) == 3:
                sensor = int(sensor_match)
            instrument = MappableInstrumentShortName(instrument_short_name)
    instrument = MappableInstrumentShortName(instrument_short_name)
    return instrument, sensor


def parse_map_duration(
    duration: str | int | timedelta,
) -> str:
    """
    Parse the duration into a string representation.

    Parameters
    ----------
    duration : str | int | timedelta
        The duration to parse. This can be a string in the format "1yr", "6mo", etc.,
        an integer representing the number of days, or a timedelta object.

    Returns
    -------
    str
        The parsed duration string in the format "1yr", "6mo", etc.
    """
    if isinstance(duration, timedelta):
        # Convert timedelta to str representation of number of DAYS_IN_MONTH day months
        num_months = int(duration.days // DAYS_IN_MONTH)
        duration = f"{num_months}mo"
    elif isinstance(duration, int):
        # Assume number of days and convert to DAYS_IN_MONTH-day months
        duration = f"{int(duration // DAYS_IN_MONTH)}mo"
    elif isinstance(duration, str):
        pass
    else:
        raise ValueError("Invalid duration type. Must be str, int, or timedelta.")
    # Replace 12mo with 1yr
    if duration == "12mo":
        duration = "1yr"
    return duration


def get_map_coord_frame(frame_str: str | Literal["hae",]) -> SpiceFrame:
    """
    Get the SpiceFrame object for a given frame string.

    This Frame is used for rotation from PSET coordinates to
    SkyMap coordinates, and so is more analogous the the
    coordinate frame component of the map descriptor string
    than to the hf/sf/hk frame component.

    Parameters
    ----------
    frame_str : Literal["hae",]
        The frame string corresponding to the SpiceFrame object.

    Returns
    -------
    SpiceFrame
        The SpiceFrame object corresponding to the frame string.

    Raises
    ------
    NotImplementedError
        If the frame string is not recognized.
    """
    if frame_str == "hae":
        return SpiceFrame.ECLIPJ2000
    else:
        raise NotImplementedError("Coordinate frame is not yet implemented.")


def parse_map_frame(
    frame: SpiceFrame | Literal["sf", "hf", "hk"] | str,
) -> str:
    """
    Parse the frame into a string representation.

    Parameters
    ----------
    frame : str | SpiceFrame
        The frame to parse. This can be a string in the format "sf", "hf", "hk", or
        a SpiceFrame object.

    Returns
    -------
    str
        The parsed frame string.
    """
    if isinstance(frame, SpiceFrame):
        match frame:
            case SpiceFrame.IMAP_DPS.value:
                return "sf"
            case SpiceFrame.ECLIPJ2000.value:
                return "hf"
            case _:
                raise NotImplementedError(f"Frame {frame} is not yet implemented.")
    # Handle string frame
    elif frame in ["sf", "hf", "hk"]:
        return frame
    else:
        raise ValueError(
            f"Invalid frame: {frame}. Expected 'sf', 'hf', 'hk', or a SpiceFrame."
        )


@dataclass
class MapDescriptor:
    """
    A class to represent a map descriptor for ENA maps.

    This class provides methods to parse a map descriptor string and convert it
    back into a string.

    Attributes
    ----------
    instrument : MappableInstrumentShortName
        The short name of the instrument.
    frame : str
        The frame descriptor string. (e.g. "sf", "hf", "hk").
    resolution_str : str
        The resolution string for the map (e.g. "nside128", "2deg").
    duration : str
        The duration of the map (e.g. "1yr", "6mo").
    sensor : str, optional
        The sensor identifier (e.g. "45", "90", "combined", "").
        Default is "".
    principal_data : str, optional
        The principal data type for the map (e.g. "ena", "spx", "isn").
        Default is "ena".
    species : str, optional
        The species for the map (e.g. "h", "he", "o").
        Default is "h".
    survival_corrected : str, optional
        Whether the map is survival probability corrected ("sp") or not ("nsp").
        Default is "nsp".
    spin_phase : str, optional
        The spin phase for the map (e.g. "full", "ram", "anti").
        Default is "full".
    coordinate_system : str, optional
        The coordinate system for the map (e.g. "hae", "hgi", "rc").
        Default is "hae".
    """

    instrument: MappableInstrumentShortName
    frame_descriptor: str | SpiceFrame
    resolution_str: str
    duration: str | int | timedelta
    sensor: _sensor_types = ""
    principal_data: str = "ena"
    species: str = "h"
    survival_corrected: str = "nsp"
    spin_phase: str = "full"
    coordinate_system: str = "hae"

    # Quantities parsed into strings that will fit in the descriptor
    @property
    def frame_str(self) -> str:
        """
        Get the frame's string representation. See parse_map_frame().

        Returns
        -------
        str
            The frame string representation.
        """
        return parse_map_frame(self.frame_descriptor)

    @property
    def duration_str(self) -> str:
        """
        Get the duration's string representation. See parse_map_duration().

        Returns
        -------
        str
            The duration string representation.
        """
        return parse_map_duration(self.duration)

    @property
    def instrument_descriptor(self) -> str:
        """
        Get the instrument descriptor string.

        Returns
        -------
        str
            The instrument descriptor string.
        """
        return get_instrument_descriptor(self.instrument, self.sensor)

    @classmethod
    def from_string(cls, map_descriptor: str) -> MapDescriptor:
        """
        Parse a map_descriptor string and return a MapDescriptor instance.

        The map_descriptor string is expected to follow the format:
        "instrument_descriptor-principal_data-species-frame-...cont...
        survival_corrected-spin_phase-coordinate_system-resolution_str-duration".

        Parameters
        ----------
        map_descriptor : str
            The map descriptor string to parse.

        Returns
        -------
        MapDescriptor
            An instance of the MapDescriptor class with parsed values.
        """
        parts = map_descriptor.split("-")
        if len(parts) != 9:
            raise ValueError(
                f"Invalid map_descriptor format: {map_descriptor}. Expected 9 parts."
            )
        # Extract the instrument and sensor from the first part
        instrument_sensor = parts[0]
        instrument, sensor = parse_instrument_descriptor(instrument_sensor)

        return cls(
            instrument=instrument,
            sensor=sensor,
            principal_data=parts[1],
            species=parts[2],
            frame_descriptor=parts[3],
            survival_corrected=parts[4],
            spin_phase=parts[5],
            coordinate_system=parts[6],
            resolution_str=parts[7],
            duration=parts[8],
        )

    def to_string(self) -> str:
        """
        Convert the MapDescriptor instance back into a map_descriptor string.

        Returns
        -------
        str
            The map_descriptor string in the format:
            "instrument_descriptor-principal_data-species-frame-...cont...
            survival_corrected-spin_phase-coordinate_system-resolution_str-duration".
        """
        return "-".join(
            [
                self.instrument_descriptor,
                self.principal_data,
                self.species,
                self.frame_str,
                self.survival_corrected,
                self.spin_phase,
                self.coordinate_system,
                self.resolution_str,
                self.duration_str,
            ]
        )


def build_l2_map_descriptor(
    *,
    # Basic parameters must always be passed in as keyword arguments
    instrument: MappableInstrumentShortName,  # Updated to use the enum
    frame: Literal["sf", "hf", "hk"] | SpiceFrame,
    resolution_str: str,
    duration: str | int | timedelta,
    # The rest of the parameters have default values corresponding to the
    # most general cases
    sensor: _sensor_types = "",
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
    sensor : _sensor_types
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
    instrument_descriptor = get_instrument_descriptor(instrument, sensor)
    # Handle duration
    if isinstance(duration, timedelta):
        # Convert timedelta to str representation of number of DAYS_IN_MONTH-day months
        num_months = int(duration.days // DAYS_IN_MONTH)
        duration = f"{num_months}mo"
    elif isinstance(duration, int):
        # Assume number of days and convert to DAYS_IN_MONTH-day months
        duration = f"{int(duration // DAYS_IN_MONTH)}mo"
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


def ns_to_duration_months(ns: int) -> int:
    """
    Convert nanoseconds to months using DAYS_IN_MONTH days per month approximation.

    Parameters
    ----------
    ns : int
        The number of nanoseconds to convert.

    Returns
    -------
    int
        The number of months, floored to the nearest integer.

    Notes
    -----
    This can be used to convert from the difference between two epochs in ns to the
    number of months between them.

    This is a very simple estimate, which assumes that a month is DAYS_IN_MONTH days and
    floors the result.

    This successfully yields:
    - 12 months for 365.25 days in ns
    - 6 months for 182.625 days (365.25/2) in ns
    - 4 months for 121.75 days (365.25/3) in ns
    - 3 months for 91.3125 days (365.25/4) in ns
    """
    days = ns / (1e9 * 60 * 60 * 24)
    months = days // DAYS_IN_MONTH
    return int(months)


def get_output_map_structure_from_descriptor_string(
    descriptor: str,
) -> ena_maps.HealpixSkyMap | ena_maps.RectangularSkyMap:
    """
    Get the output map structure from a map descriptor string.

    Parameters
    ----------
    descriptor : str
        The map descriptor string.

    Returns
    -------
    ena_maps.HealpixSkyMap | ena_maps.RectangularSkyMap
        The output map structure.

    Raises
    ------
    ValueError
        If the descriptor string is invalid.
    """
    map_descriptor = MapDescriptor.from_string(descriptor)

    # If "deg" is in the resolution string, then this is a rectangular map
    # (e.g., '2deg')
    if "deg" in map_descriptor.resolution_str:
        return ena_maps.RectangularSkyMap(
            spacing_deg=float(map_descriptor.resolution_str.split("deg")[0]),
            spice_frame=get_map_coord_frame(map_descriptor.coordinate_system),
        )
    # If "nside" is in the resolution string, then this is a Healpix map
    # (e.g., 'nside32')
    elif "nside" in map_descriptor.resolution_str:
        return ena_maps.HealpixSkyMap(
            nside=int(map_descriptor.resolution_str.split("nside")[1]),
            spice_frame=get_map_coord_frame(map_descriptor.coordinate_system),
        )
    else:
        raise ValueError(
            f"Could not interpret resolution string: {map_descriptor.resolution_str} "
            "as either a Healpix ('nside32') or rectangular map ('2deg')."
        )


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

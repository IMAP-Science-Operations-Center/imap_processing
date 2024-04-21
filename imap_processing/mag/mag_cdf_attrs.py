"""Shared attribute values for MAG CDF files."""

from __future__ import annotations

import dataclasses
from dataclasses import InitVar, dataclass, field
from enum import Enum

from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.cdf.global_attrs import (
    AttrBase,
    GlobalDataLevelAttrs,
    GlobalInstrumentAttrs,
    ScienceAttrs,
)
from imap_processing.mag import __version__, cdf_format_version

text = (
    "The IMAP magnetometer (MAG) consists of a pair of identical magnetometers "
    "which each measure the magnetic field in three directions in the vicinity of "
    "the spacecraft. "
    "MAG will contribute to our understanding of the acceleration and transport "
    "of charged particles in the heliosphere. "
    "MAG design and assembly is led by Imperial College, London. See "
    "https://imap.princeton.edu/instruments/mag for more details."
)


class DataMode(Enum):
    """Enum for MAG data modes: burst and normal (BURST + NORM)."""

    BURST = "BURST"
    NORM = "NORM"


class Sensor(Enum):
    """Enum for MAG sensors: raw, MAGo, and MAGi (RAW, MAGO, MAGI)."""

    MAGO = "MAGO"
    MAGI = "MAGI"
    RAW = "RAW"


@dataclass
class MagGlobalCdfAttributes:
    """
    Organize attributes for different kinds of L1A CDF files.

    Generation_date and input_files are added to the raw, mago, and magi attributes.
    The attributes are set based on the data_mode and sensor values.
    # TODO: Add tests for this

    Attributes
    ----------
    data_mode : DataMode
        The data mode of the CDF file. This is only used in __init__ and cannot be
        accessed from instances.
    sensor : Sensor
        The sensor type of the CDF file. This is only used in __init__ and cannot be
        accessed from instances.
    generation_date : str
        The date the CDF file was generated, in yyyy-mm-ddTHH:MM:SS (ISO) format.
    input_files : list[str]
        The input files used to generate the CDF file.
    attribute_dict : dict
        The attribute dictionary for the CDF file. This is not initialized in __init__
        and is set from the combination of data_mode and sensor.
    """

    data_mode: InitVar[DataMode]
    sensor: InitVar[Sensor]
    generation_date: str
    input_files: list[str]
    data_version: str
    attribute_dict: dict = field(init=False)

    def __post_init__(self, data_mode: DataMode, sensor: Sensor):
        """
        Will set the attribute dictionary attribute_dict based on data_mode and sensor.

        The data_mode and sensor class variables are not used except for selecting the
        attribute dictionary, so they are InitVar variables meaning they cannot be
        accessed as class attributes.

        Parameters
        ----------
        data_mode: DataMode
            DataMode enum
        sensor: Sensor
            Sensor enum
        """
        if data_mode == DataMode.NORM:
            if sensor == Sensor.RAW:
                self.attribute_dict = mag_l1a_norm_raw_attrs.output()
            elif sensor == Sensor.MAGO:
                self.attribute_dict = mag_l1a_norm_mago_attrs.output()
            elif sensor == Sensor.MAGI:
                self.attribute_dict = mag_l1a_norm_magi_attrs.output()
        elif data_mode == DataMode.BURST:
            if sensor == Sensor.RAW:
                self.attribute_dict = mag_l1a_burst_raw_attrs.output()
            elif sensor == Sensor.MAGO:
                self.attribute_dict = mag_l1a_burst_mago_attrs.output()
            elif sensor == Sensor.MAGI:
                self.attribute_dict = mag_l1a_burst_magi_attrs.output()

        # Add generation date and input files to the attribute dictionary
        self.attribute_dict["Generation_date"] = self.generation_date
        self.attribute_dict["Input_files"] = self.input_files
        self.attribute_dict["Data_version"] = self.data_version
        self.attribute_dict["Data_format_version"] = cdf_format_version


mag_base = GlobalInstrumentAttrs(
    # TODO: This should be the _data version_, not the package version.
    __version__,
    "MAG>Magnetometer",
    text,
    "Magnetic Fields (space)",
)

mag_data_base = GlobalDataLevelAttrs(
    "L1A_raw-norm>Level-1A raw normal rate",
    logical_source="imap_mag_l1a_norm-raw",
    logical_source_desc="IMAP Mission MAG Normal Rate Instrument Level-1A Data.",
    instrument_base=mag_base,
    additional_attrs={
        "Software_version": __version__,
        "Rules_of_use": "Not for publication",
    },
)

mag_l1a_norm_raw_attrs = mag_data_base

mag_l1a_burst_raw_attrs = dataclasses.replace(
    mag_data_base,
    data_type="L1A_raw-burst>Level-1A raw burst rate",
    logical_source="imap_mag_l1a_burst-raw",
    logical_source_desc="IMAP Mission MAG Burst Rate Instrument Level-1A Data.",
)

mag_l1a_norm_mago_attrs = dataclasses.replace(
    mag_data_base,
    data_type="L1A_raw-norm-mago>Level 1A MAGo normal rate",
    logical_source="imap_mag_l1a_norm-mago",
    logical_source_desc="IMAP Mission MAGo Normal Rate Instrument Level-1A Data.",
)

mag_l1a_norm_magi_attrs = dataclasses.replace(
    mag_data_base,
    data_type="L1A_raw-norm-magi>Level 1A MAGi normal rate",
    logical_source="imap_mag_l1a_norm-magi",
    logical_source_desc="IMAP Mission MAGi Normal Rate Instrument Level-1A Data.",
)

mag_l1a_burst_mago_attrs = dataclasses.replace(
    mag_data_base,
    data_type="L1A_raw-burst-mago>Level 1A MAGo burst rate",
    logical_source="imap_mag_l1a_burst-mago",
    logical_source_desc="IMAP Mission MAGo Burst Rate Instrument Level-1A Data.",
)

mag_l1a_burst_magi_attrs = dataclasses.replace(
    mag_data_base,
    data_type="L1A_raw-burst-magi>Level 1A MAGi burst rate",
    logical_source="imap_mag_l1a_burst-magi",
    logical_source_desc="IMAP Mission MAGi Burst Rate Instrument Level-1A Data.",
)

mag_l1b_attrs = GlobalDataLevelAttrs(
    "L1B_SCI>Level-1B Science Data",
    # TODO: replace "sci" with descriptor "norm" / "burst"
    logical_source="imap_mag_l1b_sci",
    logical_source_desc="IMAP Mission MAG Instrument Level-1B Data.",
    instrument_base=mag_base,
)

mag_l1c_attrs = GlobalDataLevelAttrs(
    "L1C_SCI>Level-1C Science Data",
    # TODO: replace "sci" with descriptor "norm" / "burst"
    logical_source="imap_mag_l1c_sci",
    logical_source_desc="IMAP Mission MAG Instrument Level-1C Data.",
    instrument_base=mag_base,
)

# TODO: Supporting data attributes?

# TODO: display type, catdesc, units, format, label_axis

# TODO: update descriptor to be more accurate for L1A raw
# TODO: does raw value need "counts"
mag_raw_vector_attrs = ScienceAttrs(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=GlobalConstants.INT_MAXVAL,
    catdesc="Raw, unprocessed magnetic field vector data in bytes",
    depend_0="epoch",
    depend_1="direction",
    display_type="time_series",
    fieldname="Magnetic Field Vector",
    label_axis="Raw binary magnetic field vector data",
    fill_val=GlobalConstants.INT_MAXVAL,
    format="I3",
    units="counts",
    var_type="data",
)

vector_attrs = ScienceAttrs(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=GlobalConstants.INT_MAXVAL,
    catdesc="Magnetic field vector with x, y, z, and sensor range, varying by time",
    depend_0="epoch",
    depend_1="direction",
    display_type="time_series",
    fieldname="Magnetic Field Vector",
    label_axis="Magnetic field vector data",
    fill_val=GlobalConstants.INT_MAXVAL,
    format="I3",
    units="counts",
    var_type="data",
)

mag_support_attrs = ScienceAttrs(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=GlobalConstants.INT_MAXVAL,
    depend_0="epoch",
    display_type="time_series",
    fill_val=GlobalConstants.INT_FILLVAL,
    format="I12",
    var_type="support_data",
)

mag_metadata_attrs = AttrBase(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=GlobalConstants.INT_MAXVAL,
    display_type="time_series",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="metadata",
)

mag_flag_attrs = ScienceAttrs(
    validmin=0,
    validmax=1,
    depend_0="epoch",
    display_type="time_series",
    fill_val=255,
    format="I1",
)

raw_direction_attrs = AttrBase(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=GlobalConstants.INT_MAXVAL,
    catdesc="Raw magnetic field vector binary length",
    fieldname="Raw magnetic field vector binary length",
    format="I3",
    var_type="support_data",
    display_type="time_series",
    label_axis="Magnetic field vector directions",
)

direction_attrs = AttrBase(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=GlobalConstants.INT_MAXVAL,
    catdesc="Magnetic field vector",
    fieldname="[x,y,z] magnetic field vector",
    format="I3",
    var_type="support_data",
    display_type="time_series",
    label_axis="Magnetic field vector",
)

# Catdesc (<80 chars), Fieldnam (<30 chars)

catdesc_fieldname_l0 = {
    "VERSION": ["CCSDS Packet Version Number", "Packet Version Number"],
    "TYPE": ["CCSDS Packet Type Indicator", "Packet Type Indicator"],
    "SEC_HDR_FLG": [
        "CCSDS Packet Secondary Header Flag",
        "Packet Secondary Header Flag",
    ],
    "PKT_APID": [
        "CCSDS Packet Application Process ID",
        "Packet Application Process ID",
    ],
    "SEQ_FLGS": ["CCSDS Packet Grouping Flags", "Packet Grouping Flags"],
    "SRC_SEQ_CTR": ["CCSDS Packet Sequence Count", "Packet Sequence Count"],
    "PKT_LEN": ["CCSDS Packet Length", "Packet Length"],
    "PUS_SPARE1": ["Spare header from ESA Standard", "Spare header"],
    "PUS_VERSION": ["PUS Version number", "PUS Version number"],
    "PUS_SPARE2": ["Spare header from ESA Standard", "Spare header"],
    "PUS_STYPE": ["PUS Service type", "PUS Service type"],
    "PUS_SSUBTYPE": [
        "PUS Service subtype, the number of seconds of data minus 1",
        "Number of seconds of data minus 1",
    ],
    "COMPRESSION": [
        "Indicates if the data is compressed, with 1 indicating the data is"
        " compressed",
        "Data is compressed",
    ],
    "MAGO_ACT": ["MAGO Active status", "MAGO Active status boolean"],
    "MAGI_ACT": ["MAGI Active status", "MAGI Active status boolean"],
    "PRI_SENS": [
        "Indicates which instrument is designated as primary. 0 is MAGo, 1 is MAGi",
        "MAGi primary sensor boolean",
    ],
    "SPARE1": ["Spare", "Spare"],
    "PRI_VECSEC": ["Primary vectors per second count", "Primary vectors per second"],
    "SEC_VECSEC": [
        "Secondary vectors per second count",
        "Secondary vectors per second",
    ],
    "SPARE2": ["Spare", "Spare"],
    "PRI_COARSETM": [
        "Primary coarse time, mission SCLK in whole seconds",
        "Primary coarse time (s)",
    ],
    "PRI_FNTM": [
        "Primary fine time, mission SCLK in 16bit subseconds",
        "Primary fine time (16 bit subsecond)",
    ],
    "SEC_COARSETM": [
        "Secondary coarse time, mission SCLK in whole seconds",
        "Secondary coarse time (s)",
    ],
    "SEC_FNTM": [
        "Secondary fine time, mission SCLK in 16bit subseconds",
        "Secondary fine time (16 bit subsecond)",
    ],
    "VECTORS": [
        "Raw binary value of MAG Science vectors before processing",
        "Raw vector binary",
    ],
}

catdesc_fieldname_l1a = {
    "is_mago": [
        "Indicates if the data is from MAGo (True is MAGo, false is MAGi)",
        "Data is from MAGo",
    ],
    "active": ["Indicates if the sensor is active", "Sensor is active"],
    # TODO is this in CDF
    "start_time": ["The coarse and fine time for the sensor", ""],
    "vectors_per_second": [
        "Number of vectors measured per second, determined by the instrument mode",
        "Vectors per second",
    ],
    "expected_vector_count": [
        "Expected number of vectors (vectors_per_second * seconds_of_data)",
        "Expected vector count",
    ],
    "seconds_of_data": ["Number of seconds of data", "Seconds of data"],
    "SHCOARSE": ["Mission elapsed time", "Mission elapsed time"],
    "vectors": [
        "List of magnetic vector samples. Each sample is in the format [x,y,z,rng] "
        "for the x, y, z coordinates of the field and the range of the instrument.",
        "Magnetic field vectors, [x, y, z, range]",
    ],
}

"""Shared attribute values for MAG CDF files."""

from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.cdf.global_attrs import (
    AttrBase,
    GlobalDataLevelAttrs,
    GlobalInstrumentAttrs,
    ScienceAttrs,
)
from imap_processing.mag import __version__

text = (
    "The IMAP magnetometer (MAG) consists of a pair of identical magnetometers "
    "which each measure the magnetic field in three directions in the vicinity of "
    "the spacecraft. "
    "MAG will contribute to our understanding of the acceleration and transport "
    "of charged particles in the heliosphere. "
    "MAG design and assembly is led by Imperial College, London. See "
    "https://imap.princeton.edu/instruments/mag for more details."
)

mag_base = GlobalInstrumentAttrs(
    __version__,
    "MAG>Magnetometer",
    text,
    "Magnetic Fields (space)",
)

mag_l1a_norm_raw_attrs = GlobalDataLevelAttrs(
    "L1A-raw-norm>Level 1A raw normal rate",
    # Should also include data type
    logical_source="imap_mag_l1a_norm-raw",
    logical_source_desc="IMAP Mission MAG Instrument Level-1A Data.",
    instrument_base=mag_base,
)

mag_l1a_burst_raw_attrs = GlobalDataLevelAttrs(
    "L1A-raw-burst>Level 1A raw burst rate",
    # Should also include data type
    logical_source="imap_mag_l1a_burst-raw",
    logical_source_desc="IMAP Mission MAG Instrument Level-1A Data.",
    instrument_base=mag_base,
)

mag_l1a_norm_mago_attrs = GlobalDataLevelAttrs(
    "L1A-raw-norm-mago>Level 1A MAGo normal rate",
    # Should also include data type
    logical_source="imap_mag_l1a_norm-mago",
    logical_source_desc="IMAP Mission MAG Instrument Level-1A Data.",
    instrument_base=mag_base,
)

mag_l1a_norm_magi_attrs = GlobalDataLevelAttrs(
    "L1A-raw-norm-magi>Level 1A MAGi normal rate",
    # Should also include data type
    logical_source="imap_mag_l1a_norm-magi",
    logical_source_desc="IMAP Mission MAG Instrument Level-1A Data.",
    instrument_base=mag_base,
)

mag_l1a_burst_mago_attrs = GlobalDataLevelAttrs(
    "L1A-raw-burst-mago>Level 1A MAGo burst rate",
    # Should also include data type
    logical_source="imap_mag_l1a_burst-mago",
    logical_source_desc="IMAP Mission MAG Instrument Level-1A Data.",
    instrument_base=mag_base,
)

mag_l1a_burst_magi_attrs = GlobalDataLevelAttrs(
    "L1A-raw-burst-magi>Level 1A MAGi burst rate",
    # Should also include data type
    logical_source="imap_mag_l1a_burst-magi",
    logical_source_desc="IMAP Mission MAG Instrument Level-1A Data.",
    instrument_base=mag_base,
)


mag_l1b_attrs = GlobalDataLevelAttrs(
    "L1A>Level-1B",
    # TODO: replace "sci" with descriptor "norm" / "burst"
    logical_source="imap_mag_l1b_sci",
    logical_source_desc="IMAP Mission MAG Instrument Level-1B Data.",
    instrument_base=mag_base,
)

mag_l1c_attrs = GlobalDataLevelAttrs(
    "L1A>Level-1C",
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
# aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

# aaaaaaaaaaaaaaaaaaaaaaaaaaaaa
# Catdesc (<80 chars), Fieldnam (<30 chars)

catdesc_fieldname_l0 = {
    # TODO: Don't include PUS_SPARE1?
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

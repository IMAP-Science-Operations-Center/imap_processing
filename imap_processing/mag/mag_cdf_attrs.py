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

mag_l1a_attrs = GlobalDataLevelAttrs(
    # TODO: data type should include "norm" and "burst" L1A-norm>Level-1A-normal-rate
    "L1A>Level-1A",
    # Should also include data type
    logical_source="imap_mag_l1a",
    logical_source_desc="IMAP Mission MAG Instrument Level-1A Data.",
    instrument_base=mag_base,
)

mag_l1b_attrs = GlobalDataLevelAttrs(
    "L1A>Level-1B",
    logical_source="imap_mag_l1b",
    logical_source_desc="IMAP Mission MAG Instrument Level-1B Data.",
    instrument_base=mag_base,
)

mag_l1c_attrs = GlobalDataLevelAttrs(
    "L1A>Level-1C",
    logical_source="imap_mag_l1c",
    logical_source_desc="IMAP Mission MAG Instrument Level-1C Data.",
    instrument_base=mag_base,
)

# TODO: Supporting data attributes?

# TODO: display type, catdesc, units, format, label_axis

# TODO: update descriptor to be more accurate for L1A raw
# TODO: does raw value need "counts"
mag_vector_attrs = ScienceAttrs(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=GlobalConstants.INT_MAXVAL,
    catdesc="Magnetic field vectors",
    depend_0="Epoch",
    depend_1="Direction",
    display_type="time_series",
    fieldname="Magnetic Field Vector",
    label_axis="Magnetic field vector",
    fill_val=GlobalConstants.INT_MAXVAL,
    format="I3",
    units="counts",
    var_type="data",
)

mag_support_attrs = ScienceAttrs(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=GlobalConstants.INT_MAXVAL,
    depend_0="Epoch",
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
    depend_0="Epoch",
    display_type="time_series",
    fill_val=255,
    format="I1",
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

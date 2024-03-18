"""Shared attribute values for IMAP-Hi CDF files."""
from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.cdf.global_attrs import (
    AttrBase,
    GlobalDataLevelAttrs,
    GlobalInstrumentAttrs,
)
from imap_processing.hi import __version__

text = (
    "IMAP-Hi consists of two identical, single-pixel"
    "high-energy energetic neutral atom (ENA) imagers"
    "mounted at fixed angles of 90 and 45 degrees"
    "relative to the spacecraft spin axis. These "
    "imagers measure neutral atoms entering our solar "
    "system from the outer edge of the heliosphere as "
    "they move towards the Sun. The ENA imagers collect"
    " the neutral atoms, sort them by type, and then map "
    "their incident direction from the outer heliosphere. "
    "IMAP-Hi uses a time-of-flight (TOF) section to "
    "identify hydrogen (H) and helium (He) and heavier "
    "atoms such as carbon (C), nitrogen (N), oxygen (O), "
    "and neon (Ne). With each spin of the spacecraft, the "
    "imagers sample swaths in the sky that include the "
    "ecliptic poles and four additional locations in the "
    "ecliptic plane. Some low latitude regions, that "
    "contain ENA emissions from the nose and tail of the "
    "heliosphere, as well as most of the IBEX Ribbon and "
    "Belt, are sampled twice within as little as 1.5 "
    "months which allows it to explore short ENA "
    "variability."
    "IMAP-Hi's design and assembly is led by Los Alamos "
    "National Laboratory (LANL) in collaboration with "
    "Southwest Research Institute (SwRI), University of "
    "New Hampshire (UNH), and University of Bern (UBe)."
    "See https://imap.princeton.edu/instruments/hi for "
    "more details."
)

hi_base = GlobalInstrumentAttrs(
    __version__,
    "IMAP-Hi>IMAP High-Energy Energetic Neutral Atom Imager",
    text,
    "Particles (space)",
)

# Direct event attrs
# TODO: combine these two based on feedback from Paul
met_subseconds_attrs = AttrBase(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=1023,
    display_type="time_series",
    catdesc=(
        "Integer millisecond of MET(aka subseconds). "
        "It's a 10-bits integer value that represents the "
        "subseconds of the MET time. Max value is 1023."
    ),
    fieldname="MET subseconds",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="metadata",
)

met_seconds_attrs = AttrBase(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=GlobalConstants.INT_MAXVAL,
    display_type="time_series",
    catdesc=(
        "integer MET(seconds). "
        "It's a 32-bits integer value that represents the "
        "seconds of the MET time. Max value is 2^32-1."
    ),
    fieldname="MET seconds",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="metadata",
)

esa_step_attrs = AttrBase(
    validmin=0,
    validmax=10,
    display_type="time_series",
    catdesc=(
        "ESA step. "
        "It's a 4-bits integer value that represents the "
        "ESA step. ESA step value range from 0-10. "
        "nominally 9 ESA steps, but possibly 8 or 10. "
        "'Step 0' is likely to refer to a background test."
    ),
    fieldname="ESA step",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="metadata",
)

de_tag_attrs = AttrBase(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=GlobalConstants.INT_MAXVAL,
    display_type="time_series",
    catdesc=(
        "Direct event tag. "
        "It's a 16-bits integer value that represents the "
        "direct event time tag."
    ),
    fieldname="Direct event tag",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="metadata",
)

trigger_id_attrs = AttrBase(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=3,
    display_type="time_series",
    catdesc=(
        "Trigger ID is a 2-bits. It represents the trigger "
        "which detector was hit first. It can be 1, 2, 3 for "
        "detector A, B, C respectively."
    ),
    fieldname="Trigger ID",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="metadata",
)

tof_attrs = AttrBase(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=1023,
    display_type="time_series",
    catdesc=(
        "Time of flight is 10-bits integer value that represents "
        "the time of flight of the direct event. 1023 is the value"
        " used to indicate no event was registered."
    ),
    fieldname="Time of flight",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="metadata",
)

# Note from SPDF about Logical_source_id breakdown:
# source_name: imap - This is in global attributes
# descriptor: instrument name - This is in global attributes
# data_type: <data_level>_<descriptor> - this is here in DataLevelAttrs
hi_de_l1a_attrs = GlobalDataLevelAttrs(
    data_type="L1A>l1a_de",
    logical_source="imap_hi_l1a_de",
    logical_source_desc=("IMAP-HI Instrument Level-1A Direct Event Data."),
    instrument_base=hi_base,
)

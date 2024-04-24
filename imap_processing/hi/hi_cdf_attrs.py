"""Shared attribute values for IMAP-Hi CDF files."""

from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.cdf.global_attrs import (
    AttrBase,
    GlobalDataLevelAttrs,
    GlobalInstrumentAttrs,
    ScienceAttrs,
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
    version=__version__,
    descriptor="Hi>IMAP High-Energy (IMAP-Hi) Energetic Neutral Atom Imager",
    text=text,
    instrument_type="Particles (space)",
)

# Direct event attrs
esa_step_attrs = ScienceAttrs(
    validmin=0,
    validmax=10,
    format="I2",
    label_axis="ESA step",
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
    depend_0="epoch",
)

de_tag_attrs = ScienceAttrs(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=GlobalConstants.INT_MAXVAL,
    format="I6",
    label_axis="Direct event time tag",
    display_type="time_series",
    catdesc=(
        "Direct event tag. "
        "It's a 16-bits integer value that represents the "
        "direct event time tag."
    ),
    fieldname="Direct event tag",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="metadata",
    depend_0="epoch",
)

trigger_id_attrs = ScienceAttrs(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=3,
    format="I1",
    label_axis="Trigger ID",
    display_type="time_series",
    catdesc=(
        "Trigger ID is a 2-bits. It represents the trigger "
        "which detector was hit first. It can be 1, 2, 3 for "
        "detector A, B, C respectively."
    ),
    fieldname="Trigger ID",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="metadata",
    depend_0="epoch",
)

tof_attrs = ScienceAttrs(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=1023,
    format="I4",
    label_axis="Time of flight",
    display_type="time_series",
    catdesc=(
        "Time of flight is 10-bits integer value that represents "
        "the time of flight of the direct event. 1023 is the value"
        " used to indicate no event was registered."
    ),
    fieldname="Time of flight",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="metadata",
    depend_0="epoch",
)

ccsds_met_attrs = ScienceAttrs(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=GlobalConstants.INT_MAXVAL,
    format="I12",
    label_axis="CCSDS MET",
    display_type="time_series",
    catdesc=(
        "CCSDS MET. "
        "It's a 32-bits integer value that represents the "
        "CCSDS Mission Elapsed Time (MET) in seconds."
    ),
    fieldname="CCSDS MET",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="metadata",
    depend_0="epoch",
)

# Note from SPDF about Logical_source_id breakdown:
# data_type: <data_level>_<descriptor> - this is here in DataLevelAttrs
hi_de_l1a_attrs = GlobalDataLevelAttrs(
    data_type="L1A_DE>Level-1A Direct Event",
    logical_source="imap_hi_l1a_de",
    logical_source_desc=("IMAP-Hi Instrument Level-1A Direct Event Data."),
    instrument_base=hi_base,
)

hi_hk_l1a_attrs = GlobalDataLevelAttrs(
    data_type="L1A_HK>Level-1A Housekeeping",
    logical_source="imap_hi_l1a_hk",
    logical_source_desc=("IMAP-Hi Instrument Level-1A Housekeeping Data."),
    instrument_base=hi_base,
)

hi_hk_l1a_metadata_attrs = ScienceAttrs(
    validmin=0,
    validmax=GlobalConstants.INT_MAXVAL,
    depend_0="epoch",
    format="I12",
    units="int",
    var_type="support_data",
    variable_purpose="PRIMARY",
)

# Histogram attributes
hi_hist_l1a_global_attrs = GlobalDataLevelAttrs(
    data_type="L1A_CNT>Level-1A Histogram",
    logical_source="imap_hi_l1a_hist",
    logical_source_desc=("IMAP-Hi Instrument Level-1A Histogram Data."),
    instrument_base=hi_base,
)

hi_hist_l1a_angle_attrs = AttrBase(
    validmin=0,
    validmax=360,
    format="I3",
    units="deg",
    label_axis="ANGLE",
    display_type="time_series",
    catdesc="Angle bin centers for Histogram data.",
    fieldname="ANGLE",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="support_data",
)

hi_hist_l1a_esa_step_attrs = ScienceAttrs(
    validmin=0,
    validmax=2**4 - 1,
    depend_0="epoch",
    format="I2",
    catdesc="4-bit ESA Step Number",
    var_type="support_data",
    variable_purpose="PRIMARY",
)

hi_hist_l1a_counter_attrs = ScienceAttrs(
    validmin=0,
    validmax=2**12 - 1,
    depend_0="epoch",
    depend_1="angle",
    format="I4",
    var_type="support_data",
    variable_purpose="PRIMARY",
)

"""Shared attribute values for IMAP-Hi CDF files."""

from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.cdf.global_attrs import (
    GlobalDataLevelAttrs,
    GlobalInstrumentAttrs,
    ScienceAttrs,
)
from imap_processing.hi import __version__

# TODO: add more information about dataset
text = (
    "IMAP-Hi consists of two identical, single-pixel"
    "high-energy energetic neutral atom (ENA) imagers"
    "mounted at fixed angles of 90 and 45 degrees"
    "relative to the spacecraft spin axis. These "
    "imagers measure neutral atoms entering our solar "
    "system from the outer edge of the heliosphere as "
    "they move towards the Sun. "
    "See https://imap.princeton.edu/instruments/imap-hi for "
    "more details. "
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
    label_axis="Energy Step",
    display_type="time_series",
    catdesc=(
        "ESA step (0-10), nominally 9, but possibly 8 or 10,"
        " 0 is likely a background test"
    ),
    var_notes=(
        "ESA (electrostatic analyzer) step. "
        "It's a 4-bits integer value that represents the "
        "ESA step. ESA step value range from 0-10. "
        "nominally 9 ESA steps, but possibly 8 or 10. "
        "'Step 0' is likely to refer to a background test. "
        "This value is used to look up the actual energy "
        "value from its lookup table."
    ),
    fieldname="ESA step number (0-10)",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="support_data",
    depend_0="epoch",
)

de_tag_attrs = ScienceAttrs(
    validmin=0,
    validmax=GlobalConstants.UBIT16_MAXVAL,
    format="I5",
    label_axis="Time Tag",
    display_type="time_series",
    catdesc=("Direct event time tag value"),
    var_notes=(
        "Direct event tag. "
        "It's a 16-bits integer value that represents the "
        "direct event time tag."
    ),
    fieldname="Direct Event Time Tag",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="support_data",
    depend_0="epoch",
)

trigger_id_attrs = ScienceAttrs(
    validmin=1,
    validmax=3,
    format="I1",
    label_axis="ID",
    display_type="time_series",
    catdesc=(
        "Trigger ID of which detector was hit first. 1,2,3 for detector A,B,C resp"
    ),
    var_notes=(
        "Trigger ID is a 2-bits. It represents the trigger "
        "which detector was hit first. It can be 1, 2, 3 for "
        "detector A, B, C respectively."
    ),
    fieldname="Trigger ID",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="data",
    depend_0="epoch",
)

tof_attrs = ScienceAttrs(
    validmin=0,
    validmax=1023,
    format="I4",
    label_axis="<replaced-in-code>",
    display_type="time_series",
    catdesc="Time of flight of direct event, 1023 indicates no event registered",
    var_notes=(
        "Time of flight is 10-bits integer value that represents "
        "the time of flight of the direct event. 1023 is the value"
        " used to indicate no event was registered."
    ),
    fieldname="<replaced-in-code>",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="data",
    depend_0="epoch",
)

# TODO: find out what is min MET time for this mission.
# This is different from the min epoch time.
ccsds_met_attrs = ScienceAttrs(
    validmin=0,
    validmax=GlobalConstants.UBIT32_MAXVAL,
    format="I10",
    label_axis="MET",
    display_type="time_series",
    catdesc="CCSDS mission elapsed time (MET)",
    var_notes=(
        "CCSDS mission elapsed time (MET). "
        "It's a 32-bits integer value that represents the "
        "CCSDS Mission Elapsed Time (MET) in seconds."
    ),
    fieldname="Mission Elapse Time(MET)",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="support_data",
    depend_0="epoch",
    units="seconds",
)

# Note from SPDF about Logical_source_id breakdown:
# data_type: <data_level>_<descriptor> - this is here in DataLevelAttrs
hi_de_l1a_attrs = GlobalDataLevelAttrs(
    data_type="L1A_DE>Level-1A Direct Event",
    logical_source="imap_hi_l1a_de",
    logical_source_desc=("IMAP-HI Instrument Level-1A Direct Event Data."),
    instrument_base=hi_base,
)

hi_hk_l1a_attrs = GlobalDataLevelAttrs(
    data_type="L1A_HK>Level-1A Housekeeping",
    logical_source="imap_hi_l1a_hk",
    logical_source_desc=("IMAP-HI Instrument Level-1A Housekeeping Data."),
    instrument_base=hi_base,
)

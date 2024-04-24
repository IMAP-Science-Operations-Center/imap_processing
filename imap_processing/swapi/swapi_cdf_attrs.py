"""Shared attribute values for SWAPI CDF files."""

from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.cdf.global_attrs import (
    AttrBase,
    GlobalDataLevelAttrs,
    GlobalInstrumentAttrs,
    ScienceAttrs,
)
from imap_processing.swapi import __version__

text = (
    "The Solar Wind and Pickup Ion (SWAPI) instrument measures several different "
    "elements of the solar wind, including hydrogen (H) and helium (He) ions, "
    "and, on occasion, heavy ions produced by large events from the Sun. See "
    "https://imap.princeton.edu/instruments/swapi for more details. SWAPI level-1 "
    "data contains primary, secondary, coincidence counts per ESA voltage step and "
    "time. Level-2 data contains the same data as level-1 but counts are converted "
    "to rates by dividing counts by time."
)

swapi_base = GlobalInstrumentAttrs(
    __version__,
    "SWAPI>The Solar Wind and Pickup Ion",
    text,
    "Particles (space)",
)

# dimensions attributes
energy_dim_attrs = AttrBase(
    validmin=0,
    validmax=72,
    format="I2",
    var_type="support_data",
    fieldname="Energy Step",
    catdesc="Energy step id in lookup table",
    label_axis="Energy Step",
)

# science data attributes
counts_attrs = ScienceAttrs(
    validmin=0,
    validmax=GlobalConstants.UINT16_MAXVAL,
    format="I5",
    units="counts",
    label_axis="<replaced-in-code>",
    display_type="spectrogram",
    catdesc="<replaced-in-code>",
    fieldname="<replaced-in-code>",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="data",
    depend_0="epoch",
    depend_1="energy",
)

# Uncertainty data is in Float and requires a different fill value,
# format, min value.
uncertainty_attrs = ScienceAttrs(
    validmin=0.0,
    validmax=GlobalConstants.FLOAT_MAXVAL,
    format="E19.5",
    label_axis="<replaced-in-code>",
    display_type="spectrogram",
    catdesc="<replaced-in-code>",
    fieldname="<replaced-in-code>",
    fill_val=GlobalConstants.DOUBLE_FILLVAL,
    var_type="data",
    depend_0="epoch",
    depend_1="energy",
    units="counts",
)

compression_attrs = ScienceAttrs(
    validmin=0,
    validmax=1,
    format="I1",
    label_axis="<replaced-in-code>",
    display_type="spectrogram",
    catdesc="<replaced-in-code>",
    var_notes=("Data compression flag. 0 if no compression, 1 if compressed"),
    fieldname="<replaced-in-code>",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="data",
    depend_0="epoch",
    depend_1="energy",
)

swapi_l1_sci_attrs = GlobalDataLevelAttrs(
    data_type="L1_sci-1min>Level-1 Science data in 1 minute resolution",
    logical_source="imap_swapi_l1_sci-1min",
    logical_source_desc=(
        "SWAPI Instrument Level-1 Science Data in 1 minute resolution."
    ),
    instrument_base=swapi_base,
)

swapi_l1_hk_attrs = GlobalDataLevelAttrs(
    data_type="L1_hk>Level-1 housekeeping data",
    logical_source="imap_swapi_l1_hk",
    logical_source_desc=("SWAPI Instrument Level-1 housekeeping Data"),
    instrument_base=swapi_base,
)

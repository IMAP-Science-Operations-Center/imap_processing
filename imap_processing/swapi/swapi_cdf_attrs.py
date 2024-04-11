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
    "The Solar Wind and Pickup Ion (SWAPI) instrument "
    "measures several different elements of the solar "
    "wind, including hydrogen (H) and helium (He) ions,"
    " and, on occasion, heavy ions produced by large "
    "events from the Sun. SWAPI also measures "
    "interstellar pickup ions (PUIs), whose material "
    "comes from beyond our solar system and moves along"
    " with the solar wind. Its data will provide "
    "information about the local ion conditions, such "
    "as temperature, density, and speed. This will also"
    " be used in the I-ALiRT data stream allowing space"
    " weather to be measured in real-time. The data from"
    " SWAPI will be valuable for understanding how the "
    "solar wind changes in response to the Sun's behavior"
    " over time. SWAPI's first-ever high time resolution "
    "measurements of helium PUIs will provide new insights"
    " into physical processes that accelerate charged "
    "particles and shape and change our global heliosphere."
    "See https://imap.princeton.edu/instruments/swapi for "
    "more details."
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
    catdesc="Energy step id in lookup table.",
    label_axis="Energy Step",
)

# science data attributes
counts_attrs = ScienceAttrs(
    validmin=0,
    validmax=GlobalConstants.INT_MAXVAL,
    format="I12",
    units="Counts",
    label_axis="Counts",
    display_type="spectrogram",
    catdesc=("Primary, secondary CEM or coincidence counts."),
    fieldname="Counts",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="data",
    depend_0="epoch",
    depend_1="energy",
)

# Uncertainty data is in Float and requires a different fill value,
# format, min value.
uncertainty_attrs = ScienceAttrs(
    validmin=GlobalConstants.DOUBLE_FILLVAL,
    validmax=GlobalConstants.INT_MAXVAL,
    format="E13.5",
    label_axis="Uncertainty",
    display_type="spectrogram",
    catdesc=("Uncertainty in the counts."),
    fieldname="Counts Uncertainty",
    fill_val=GlobalConstants.DOUBLE_FILLVAL,
    var_type="data",
    depend_0="epoch",
    depend_1="energy",
)

compression_attrs = ScienceAttrs(
    validmin=0,
    validmax=1,
    format="I1",
    label_axis="Flag",
    display_type="spectrogram",
    catdesc=("Data compression flag. 0 if no compression, 1 if compressed."),
    fieldname="Compression Flag",
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

"""Shared attribute values for ULTRA CDF files."""

from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.cdf.global_attrs import (
    AttrBase,
    GlobalDataLevelAttrs,
    GlobalInstrumentAttrs,
    ScienceAttrs,
    StringAttrs,
)
from imap_processing.ultra import __version__

descriptor = "ULTRA"
ultra_description_text = (
    "Ultra captures images of very energetic neutral atoms, "
    "particularly hydrogen (H) atoms, produced in the solar system "
    "at the heliosheath, the region where the solar wind slows, "
    "compresses, and becomes much hotter as it meets the "
    "interstellar medium (ISM). The instrument also measures the "
    "distribution of solar wind electrons and protons, and the "
    "magnetic field. See"
    "https://imap.princeton.edu/instruments/ultra for more details."
)

ultra_base = GlobalInstrumentAttrs(
    version=__version__, descriptor=descriptor, text=ultra_description_text
)

ultra_l1a_attrs = GlobalDataLevelAttrs(
    data_type="L1A_SCI>Level-1A Science Data",
    # TODO: update descriptor "sci" field with proper descriptor
    logical_source="imap_ultra_l1a_sci",
    logical_source_desc="IMAP Mission ULTRA Instrument Level-1A Data",
    instrument_base=ultra_base,
)

ultra_support_attrs = ScienceAttrs(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=GlobalConstants.INT_MAXVAL,
    display_type="time_series",
    fill_val=GlobalConstants.INT_FILLVAL,
    format="I12",
    var_type="support_data",
    label_axis="none",
    depend_0="epoch",
)

ultra_metadata_attrs = AttrBase(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=GlobalConstants.INT_MAXVAL,
    display_type="time_series",
    fill_val=GlobalConstants.INT_FILLVAL,
    format="I12",
    label_axis="none",
    var_type="metadata",
)

# Required attrs for string data type,
# meaning array with string.
string_base = StringAttrs(
    depend_0="epoch",
)

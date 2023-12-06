# TODO: link to CDF data and move doc from swe_cdf_attrs.py to readthedocs

from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.cdf.global_attrs import (
    GlobalDataLevelAttrs,
    GlobalInstrumentAttrs,
    ScienceAttrs,
)
from imap_processing.mag import __version__

mag_base = GlobalInstrumentAttrs(__version__, "", "")

mag_l1a_attrs = GlobalDataLevelAttrs(
    "L1A>Level-1A",
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
mag_l1a_science_attrs = ScienceAttrs(
    validmin=0,
    validmax=GlobalConstants.INT_MAXVAL,
    var_type="data",
    depend_0="Epoch",
    variable_purpose="PRIMARY",
)

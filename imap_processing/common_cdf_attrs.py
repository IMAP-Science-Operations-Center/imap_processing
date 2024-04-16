"""Common CDF attributes for all CDFs."""

from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.cdf.global_attrs import (
    ScienceAttrs,
)

metadata_attrs = ScienceAttrs(
    validmin=0,
    validmax=GlobalConstants.INT_MAXVAL,
    depend_0="epoch",
    format="I12",
    units="int",
    var_type="support_data",
    variable_purpose="PRIMARY",
)

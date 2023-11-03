"""CDF attrs explanation.

This website provides information about what variables are
required and what their value should be.

https://spdf.gsfc.nasa.gov/istp_guide/istp_guide.html

In general, these are the information required to add to
xr.Dataset.attrs:
    - Project wide information such as mission name,
      institude name, PI, and such. Eg.
        "Project": ["STP>Solar-Terrestrial Physics"],
        "Source_name": ["IMAP>Interstellar Mapping and Acceleration Probe"],
        "Discipline": ["Solar Physics>Heliospheric Physics"],
        "PI_name": ["Dr. David J. McComas"],
        "PI_affiliation": [
            "Princeton Plasma Physics Laboratory",
            "100 Stellarator Road, Princeton, NJ 08540",
        ],
        "Instrument_type": ["Particles (space)"],
        "Mission_group": ["IMAP>Interstellar Mapping and Acceleration Probe"],

    - Then, we add instrument specific information such as
        "Data_type": ["L1>Level-1"],
        "Data_version": __version__,
        "Descriptor": ["SWE>Solar Wind Electron"],
        "TEXT": [
            (
                "The Solar Wind Electron (SWE) instrument measures the 3D "
                "distribution of highly energetic electrons found in the solar "
                "wind. SWE is optimized to measure in situ solar wind electrons "
                "at L1 to provide context for the ENA measurements and perform  "
                "the in situ solar wind observations necessary to understand the"
                "local structures that can affect acceleration and transport of "
                "these particles. See"
                "https://imap.princeton.edu/instruments/swe for more details."
            )
        ],
        "Logical_file_id": "FILL ME IN AT FILE CREATION",
        "Logical_source": "imap_swe_l1a",
        "Logical_source_description": ["IMAP Mission SWE Instrument Level-1 Data"],


Then we create xr.Dataset and add xr.DataArray to the xr.Dataset. xr.DataArray stores
dimension information. xr.DataArray could be one or two or three dimension.
Based on that, the attrs setup could be different. Let's walk through some
scenario:

One of the ISTP requirement is to have 'Epoch' as first dimension.
Any additional dimension, such as 'Energy' and 'Counts', we should create
xr.DataArray with some default value and add its attributes. Then
we add it to xr.Dataset as coordinates. Note that dimension's default
value or data should be 1D array. Dimension's dims could be itself.
Eg.
    energy = xr.DataArray(
        np.arange(180),
        name="Energy",
        dims=["Energy"],
        attrs=int_attrs,
    )

Now, if xr.DataArray uses two demensions such as Epoch and Energy,
eg.
    xr.DataArray(data, dims=["Epoch", "Energy"])
Then your attrs would add this in addition to basic required attrs:
    "DEPEND_0": "Epoch",
    "DEPEND_1": "Energy",

Now if xr.DataArray uses three demensions such as Epoch, Energy, and
Counts, eg.
    xr.DataArray(data, dims=["Epoch", "Energy", "Counts"])
Then your attrs would add this in addition to basic required attrs:
    "DEPEND_0": "Epoch",
    "DEPEND_1": "Energy",
    "DEPEND_2": "Counts",


About FORMAT data types
=======================
"A" and "I" represent the data type
(Alphanumeric and integer, respectively), and the number
is how many characters to use to display.  So A80 means
use up to 80 characters to display the string, and I12
means use up to 12 characters to display the integer

Per SPDF contacts, we can ignore this warning:
Warning: CDF is set for row major array variables and column major is recommended.
"""

import numpy as np

from imap_processing import cdf_utils
from imap_processing.swe import __version__

# Global Attributes for different level
swe_global_base = cdf_utils.global_base.copy()
swe_global_base.update(
    {
        "Data_version": __version__,
        "Descriptor": ["SWE>Solar Wind Electron"],
        "TEXT": [
            (
                "The Solar Wind Electron (SWE) instrument measures the 3D "
                "distribution of highly energetic electrons found in the solar "
                "wind. SWE is optimized to measure in situ solar wind electrons "
                "at L1 to provide context for the ENA measurements and perform  "
                "the in situ solar wind observations necessary to understand the"
                "local structures that can affect acceleration and transport of "
                "these particles. See"
                "https://imap.princeton.edu/instruments/swe for more details."
            )
        ],
        "Logical_file_id": "FILL ME IN AT FILE CREATION",
    }
)


# Required attrs for data type of
# number, int and float.
# We can create one attrs dict for both
# with their default values.

# For SWE, max range for uncompressed counts is
# 65536. Therefore, I initialized it with that
# value.
int_attrs = {
    "CATDESC": None,
    "DISPLAY_TYPE": "spectrogram",
    "FIELDNAM": None,
    "FILLVAL": cdf_utils.INT_FILLVAL,
    "FORMAT": "I18",
    "LABLAXIS": None,
    "UNITS": "int",
    "VALIDMIN": np.int64(0),
    "VALIDMAX": np.int64(65536),
    "VAR_TYPE": "support_data",
    "SCALETYP": "linear",
}

# For SWE, max range for uncompressed rates is
# less than 800,000 when uncompressed count is
# divided by 0.083 second.
# TODO: check with SWE team if they want to keep
# range in millieseconds or seconds and update this
# max range accordingly.
float_attrs = {
    "CATDESC": None,
    "DISPLAY_TYPE": "spectrogram",
    "FIELDNAM": None,
    "FILLVAL": cdf_utils.DOUBLE_FILLVAL,
    "FORMAT": "F64.5",
    "LABLAXIS": None,
    "UNITS": "float",
    "VALIDMIN": np.float64(-800000),
    "VALIDMAX": np.float64(800000),
    "VAR_TYPE": "support_data",
    "SCALETYP": "linear",
}

# Housekeeping mode data array is stored as string.
# Required attrs for string data type,
# meaning array with string.
string_attrs = {
    "CATDESC": None,
    "DEPEND_0": "Epoch",
    "FORMAT": "A80",
    "DISPLAY_TYPE": "no_plot",
    "FIELDNAM": None,
    "VAR_TYPE": "metadata",
}


# For each data level, create global attrs
swe_l1a_global_attrs = swe_global_base.copy()
swe_l1a_global_attrs.update(
    {
        "Data_type": "L1A>Level-1A",
        "Logical_source": "imap_swe_l1a",
        "Logical_source_description": "IMAP Mission SWE Instrument Level-1A Data.",
    }
)


swe_l1b_global_attrs = swe_global_base.copy()
swe_l1b_global_attrs.update(
    {
        "Data_type": "L1B>Level-1B",
        "Logical_source": "imap_swe_l1b",
        "Logical_source_description": "IMAP Mission SWE Instrument Level-1B Data.",
    }
)


l1a_science_attrs = {
    "CATDESC": "Science Data",
    "DEPEND_0": "Epoch",
    "DEPEND_1": "Energy",
    "DEPEND_2": "Counts",
    "DISPLAY_TYPE": "time_series",
    "FIELDNAM": "Counts",
    "FILLVAL": cdf_utils.INT_FILLVAL,
    "FORMAT": "I18",
    "LABLAXIS": "Counts",
    "UNITS": "dN",
    "VALIDMIN": np.int64(0),
    "VALIDMAX": np.int64(65536),
    "VAR_TYPE": "support_data",
    "SCALETYP": "linear",
}

l1b_science_attrs = {
    "CATDESC": "Science Data",
    "DEPEND_0": "Epoch",
    "DEPEND_1": "Energy",
    "DEPEND_2": "Angle",
    "DEPEND_3": "Rates",
    "DISPLAY_TYPE": "time_series",
    "FIELDNAM": "Counts",
    "FILLVAL": cdf_utils.DOUBLE_FILLVAL,
    "FORMAT": "F64.5",
    "LABLAXIS": "Counts",
    "UNITS": "dN",
    "VALIDMIN": np.float64(-800000),
    "VALIDMAX": np.float64(800000),
    "VAR_TYPE": "support_data",
    "SCALETYP": "linear",
}

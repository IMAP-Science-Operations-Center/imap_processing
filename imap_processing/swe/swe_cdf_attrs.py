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


from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.cdf.global_attrs import (
    AttrBase,
    GlobalDataLevelAttrs,
    GlobalInstrumentAttrs,
    ScienceAttrs,
    StringAttrs,
)
from imap_processing.swe import __version__

descriptor = "SWE>Solar Wind Electron"
swe_description_text = (
    "The Solar Wind Electron (SWE) instrument measures the 3D "
    "distribution of highly energetic electrons found in the solar "
    "wind. SWE is optimized to measure in situ solar wind electrons "
    "at L1 to provide context for the ENA measurements and perform  "
    "the in situ solar wind observations necessary to understand the"
    "local structures that can affect acceleration and transport of "
    "these particles. See"
    "https://imap.princeton.edu/instruments/swe for more details."
)

swe_base = GlobalInstrumentAttrs(
    version=__version__, descriptor=descriptor, text=swe_description_text
)


swe_l1a_global_attrs = GlobalDataLevelAttrs(
    data_type="L1A->Level-1A",
    logical_source="imap_swe_l1a",
    logical_source_desc="IMAP Mission SWE Instrument Level-1A Data",
    instrument_base=swe_base,
)

swe_l1b_global_attrs = GlobalDataLevelAttrs(
    data_type="L1B->Level-1B",
    logical_source="imap_swe_l1b",
    logical_source_desc="IMAP Mission SWE Instrument Level-1B Data",
    instrument_base=swe_base,
)

# Uses TypeBase because because IntBase and FloatBase
# requires depend_0 since it inherits from ScienceBase.
# Below int_base and float_base is used to defined attrs
# for coordinates systems that doesn't have depend_0.

int_base = AttrBase(
    validmin=0,
    validmax=GlobalConstants.INT_MAXVAL,
    format="I12",
    var_type="support_data",
    display_type="no_plot",
)

float_base = AttrBase(
    validmin=0,
    validmax=GlobalConstants.INT_MAXVAL,
    format="I12",
    var_type="support_data",
    display_type="no_plot",
)

# Housekeeping mode data array is stored as string.
# Required attrs for string data type,
# meaning array with string.
string_base = StringAttrs(
    depend_0="Epoch",
)

swe_metadata_attrs = ScienceAttrs(
    validmin=0,
    validmax=GlobalConstants.INT_MAXVAL,
    display_type="no_plot",
    depend_0="Epoch",
    format="I12",
    units="dN",
    var_type="support_data",
    variable_purpose="PRIMARY",
)

# TODO: ask SWE team about valid min and max values of
# these data
l1a_science_attrs = ScienceAttrs(
    validmin=0,
    validmax=GlobalConstants.INT_MAXVAL,
    display_type="spectrogram",
    depend_0="Epoch",
    depend_1="Energy",
    depend_2="Counts",
    format="I12",
    units="dN",
    var_type="data",
    variable_purpose="PRIMARY",
)

l1b_science_attrs = ScienceAttrs(
    validmin=0,
    validmax=GlobalConstants.INT_MAXVAL,
    display_type="spectrogram",
    depend_0="Epoch",
    depend_1="Energy",
    depend_2="Angle",
    depend_3="Rates",
    format="I12",
    units="dN",
    var_type="data",
    variable_purpose="PRIMARY",
)

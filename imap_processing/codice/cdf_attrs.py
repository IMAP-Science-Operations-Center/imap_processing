"""CDF attrs for CoDICE.

This website provides information about what variables are required and what
their value should be:

https://spdf.gsfc.nasa.gov/istp_guide/istp_guide.html

For further details, see the documentation provided at
https://imap-processing.readthedocs.io/en/latest/development/CDFs/cdf_requirements.html
"""


from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.cdf.global_attrs import (
    AttrBase,
    GlobalDataLevelAttrs,
    GlobalInstrumentAttrs,
    ScienceAttrs,
    StringAttrs,
)
from imap_processing.codice import __version__

descriptor = "CoDICE>Compact Dual Ion Composition Experiment"
codice_description_text = (
    "The Compact Dual Ion Composition Experiment (CoDICE) will measure the "
    "distributions and composition of interstellar pickup ions (PUIs), "
    "particles that make it through the heliosheath into the heliosphere. "
    "CoDICE also collects and characterizes solar wind ions including the "
    "mass and composition of highly energized particles (called suprathermal) "
    "from the Sun. CoDICE combines an electrostatic analyzer(ESA) with a "
    "Time-Of-Flight versus Energy (TOF / E) subsystem to simultaneously  "
    "measure the velocity, arrival direction, ionic charge state, and mass of "
    "specific species of ions in the LISM. CoDICE also has a path for higher "
    "energy particles to skip the ESA but still get measured by the common "
    "TOF / E system. These measurements are critical in determining the Local "
    "Interstellar Medium (LISM) composition and flow properties, the origin of "
    "the enigmatic suprathermal tails on the solar wind distributions and "
    "advance understanding of the acceleration of particles in the heliosphere."
)

codice_base = GlobalInstrumentAttrs(
    version=__version__, descriptor=descriptor, text=codice_description_text
)

codice_l1a_global_attrs = GlobalDataLevelAttrs(
    data_type="L1A->Level-1A",
    logical_source="imap_codice_l1a",
    logical_source_desc="IMAP Mission CoDICE Instrument Level-1A Data",
    instrument_base=codice_base,
)

codice_l1b_global_attrs = GlobalDataLevelAttrs(
    data_type="L1B->Level-1B",
    logical_source="imap_cpdice_l1b",
    logical_source_desc="IMAP Mission CoDICE Instrument Level-1B Data",
    instrument_base=codice_base,
)

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

string_base = StringAttrs(
    depend_0="Epoch",
)

codice_metadata_attrs = ScienceAttrs(
    validmin=0,
    validmax=GlobalConstants.INT_MAXVAL,
    display_type="no_plot",
    depend_0="Epoch",
    format="I12",
    units="dN",
    var_type="support_data",
    variable_purpose="PRIMARY",
)

# TODO: ask CoDICE team about valid min and max values of these data
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

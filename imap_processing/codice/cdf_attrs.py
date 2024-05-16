"""CDF attrs for CoDICE.

For further details, see the documentation provided at
https://imap-processing.readthedocs.io/en/latest/development/CDFs/cdf_requirements.html

Reference: https://spdf.gsfc.nasa.gov/sp_use_of_cdf.html
"""

# TODO: Add catdescs
# TODO: Validmax should be 2^24 not 2^64

from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.cdf.global_attrs import (
    AttrBase,
    GlobalDataLevelAttrs,
    GlobalInstrumentAttrs,
    ScienceAttrs,
)
from imap_processing.codice import __version__

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
    __version__,
    "CoDICE>Compact Dual Ion Composition Experiment",
    codice_description_text,
    "Particles (space)",
)

# Dataset-level attributes
l1a_hskp_attrs = GlobalDataLevelAttrs(
    data_type="L1A_hskp->Level-1A Housekeeping Data",
    logical_source="imap_codice_l1a_hskp",
    logical_source_desc="IMAP Mission CoDICE Instrument Level-1A Housekeeping Data",
    instrument_base=codice_base,
)

l1a_lo_sw_species_counts_attrs = GlobalDataLevelAttrs(
    data_type="L1A_lo-sw-species-counts->Level-1A Lo Sunward Species Counts Data",
    logical_source="imap_codice_l1a_lo-sw-species-counts",
    logical_source_desc=(
        "IMAP Mission CoDICE Instrument Level-1A Lo Sunward Species Counts Data"
    ),
    instrument_base=codice_base,
)

l1a_lo_nsw_species_counts_attrs = GlobalDataLevelAttrs(
    data_type="L1A_lo-nsw-species-counts->Level-1A Lo Non-sunward Species Counts Data",
    logical_source="imap_codice_l1a_lo-nsw-species-counts",
    logical_source_desc=(
        "IMAP Mission CoDICE Instrument Level-1A Lo Non-sunward Species Counts Data"
    ),
    instrument_base=codice_base,
)

l1a_lo_sw_priority_counts_attrs = GlobalDataLevelAttrs(
    data_type="L1A_lo-sw-priority-counts->Level-1A Lo Sunward Priority Counts Data",
    logical_source="imap_codice_l1a_lo-sw-priority-counts",
    logical_source_desc=(
        "IMAP Mission CoDICE Instrument Level-1A Lo Sunward Priority Counts Data"
    ),
    instrument_base=codice_base,
)

l1a_lo_sw_angular_counts_attrs = GlobalDataLevelAttrs(
    data_type="L1A_lo-sw-angular-counts->Level-1A Lo Sunward Angular Counts Data",
    logical_source="imap_codice_l1a_lo-sw-angular-counts",
    logical_source_desc=(
        "IMAP Mission CoDICE Instrument Level-1A Lo Sunward Angular Counts Data"
    ),
    instrument_base=codice_base,
)

# Variable-level attributes
acquisition_times_attrs = AttrBase(
    validmin=0,
    validmax=GlobalConstants.INT_MAXVAL,
    format="I12",
    var_type="support_data",
    fieldname="Acquisition Time",
    catdesc="TBD",
    label_axis="Acq Time",
)

codice_metadata_attrs = ScienceAttrs(
    validmin=0,
    validmax=GlobalConstants.INT_MAXVAL,
    display_type="no_plot",
    depend_0="epoch",
    format="I12",
    units="dN",
    var_type="data",
    variable_purpose="PRIMARY",
)

counters_attrs = ScienceAttrs(
    validmin=0,
    validmax=GlobalConstants.INT_MAXVAL,
    format="I12",
    units="counts",
    label_axis="counts",
    display_type="time_series",
    catdesc="TBD",
    fieldname="TBD",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="data",
    depend_0="epoch",
    depend_1="energy",
)

energy_attrs = AttrBase(
    validmin=0,
    validmax=127,
    format="I3",
    var_type="support_data",
    fieldname="Energy Step",
    catdesc="TBD",
    label_axis="energy",
)

esa_sweep_attrs = AttrBase(
    validmin=0,
    validmax=GlobalConstants.INT_MAXVAL,
    format="I12",
    var_type="support_data",
    fieldname="ESA V",
    catdesc="TBD",
    label_axis="ESA V",
)

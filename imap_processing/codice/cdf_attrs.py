"""CDF attrs for CoDICE.

For further details, see the documentation provided at
https://imap-processing.readthedocs.io/en/latest/development/CDFs/cdf_requirements.html

Reference: https://spdf.gsfc.nasa.gov/sp_use_of_cdf.html
"""

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

l1a_lo_sw_angular_counts_attrs = GlobalDataLevelAttrs(
    data_type="L1A_lo-sw-angular-counts->Level-1A Lo Sunward Angular Counts Data",
    logical_source="imap_codice_l1a_lo-sw-angular-counts",
    logical_source_desc=(
        "IMAP Mission CoDICE Instrument Level-1A Lo Sunward Angular Counts Data"
    ),
    instrument_base=codice_base,
)

l1a_lo_nsw_angular_counts_attrs = GlobalDataLevelAttrs(
    data_type="L1A_lo-nsw-angular-counts->Level-1A Lo Non-Sunward Angular Counts Data",
    logical_source="imap_codice_l1a_lo-nsw-angular-counts",
    logical_source_desc=(
        "IMAP Mission CoDICE Instrument Level-1A Lo Non-sunward Angular Counts Data"
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

l1a_lo_nsw_priority_counts_attrs = GlobalDataLevelAttrs(
    data_type="L1A_lo-sw-priority-counts->Level-1A Lo Non-Sunward Priority Counts Data",
    logical_source="imap_codice_l1a_lo-nsw-priority-counts",
    logical_source_desc=(
        "IMAP Mission CoDICE Instrument Level-1A Lo Non-Sunward Priority Counts Data"
    ),
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
    data_type="L1A_lo-nsw-species-counts->Level-1A Lo Non-Sunward Species Counts Data",
    logical_source="imap_codice_l1a_lo-nsw-species-counts",
    logical_source_desc=(
        "IMAP Mission CoDICE Instrument Level-1A Lo Non-Sunward Species Counts Data"
    ),
    instrument_base=codice_base,
)

# Variable-level attributes
acquisition_times_attrs = AttrBase(
    catdesc="Time of acquisition for the energy step",
    display_type="no_plot",
    fieldname="Acquisition Time",
    fill_val=GlobalConstants.DOUBLE_FILLVAL,
    format="F10.3",
    label_axis="Acq Time",
    units="ms",
    validmin=0,
    validmax=GlobalConstants.FLOAT_MAXVAL,
    var_type="support_data",
    scale_type="linear",
)

codice_metadata_attrs = ScienceAttrs(
    display_type="no_plot",
    format="I12",
    units="dN",
    validmin=0,
    validmax=GlobalConstants.INT_MAXVAL,
    var_type="data",
    variable_purpose="PRIMARY",
    depend_0="epoch",
)

# TODO: cdf.global_attrs needs to be updated to allow multiple LABL_PTRs
#       as well as to not include LABEL_AXIS when necessary. For now, hard-code
#       these so we don't need to use cdf.global_attrs.ScienceAttrs()
counters_attrs = {
    "CATDESC": "Fill in at creation",
    "DISPLAY_TYPE": "time_series",
    "FIELDNAM": "Fill in at creation",
    "FILLVAL": GlobalConstants.INT_FILLVAL,
    "FORMAT": "I12",
    "LABL_PTR_1": "energy",
    "UNITS": "counts",
    "VALIDMIN": 0,
    "VALIDMAX": 8388607,  # max value for a signed 24-bit integer
    "VAR_TYPE": "data",
    "SCALETYP": "linear",
    "DEPEND_0": "epoch",
    "DEPEND_1": "energy",
}

energy_attrs = AttrBase(
    catdesc="Energy per charge (E/q) sweeping step",
    display_type="no_plot",
    fieldname="Energy Step",
    fill_val=GlobalConstants.INT_FILLVAL,
    format="I3",
    label_axis="energy",
    units="",
    validmin=0,
    validmax=127,
    var_type="support_data",
    scale_type="linear",
)

esa_sweep_attrs = AttrBase(
    catdesc="ElectroStatic Analyzer Energy Values",
    display_type="no_plot",
    fieldname="ESA V",
    fill_val=GlobalConstants.INT_FILLVAL,
    format="I12",
    label_axis="ESA V",
    units="V",
    validmin=0,
    validmax=GlobalConstants.INT_MAXVAL,
    var_type="support_data",
    scale_type="linear",
)

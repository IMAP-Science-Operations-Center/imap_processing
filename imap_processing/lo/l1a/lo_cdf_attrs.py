# TODO: Need to add version to init
from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.cdf.global_attrs import (
    GlobalDataLevelAttrs,
    GlobalInstrumentAttrs,
    ScienceAttrs,
)
from imap_processing.lo import __version__

descriptor = "Lo>IMAP Low-Energy (IMAP-Lo) Energetic Neutral Atom Imager"
lo_description_text = (
    "IMAP-Lo is a single-pixel neutral atom imager that delivers "
    "energy and position measurements of low-energy Interstellar "
    "Neutral (ISN) atoms tracked over the ecliptic longitude >180deg "
    "and global maps of energetic neutral atoms (ENAs). Mounted on a "
    "pivot platform, IMAP-Lo tracks the flow of these ions through the "
    "local interstellar medium (LISM) to precisely determine the "
    "species-dependent flow speed, temperature, and direction of the "
    "LISM that surrounds, interacts with, and determines the outer "
    "boundaries of the global heliosphere. IMAP-Lo uses the pivoting "
    "field of view (FOV) to view variable angles out to 90deg from the "
    "spin axis. This assists IMAP-Lo to pinpoint the intersection between "
    "the ISN inflow speed and longitude to uniquely determine the LISM flow "
    "vector. Data from IMAP-Lo will help us be able to see from inside the "
    "heliosphere what it is like just outside the solar system, our local "
    "neighborhood."
)

lo_base = GlobalInstrumentAttrs(
    version=__version__,
    descriptor=descriptor,
    text=lo_description_text,
    instrument_type="Particles (space)",
)

lo_de_l1a_attrs = GlobalDataLevelAttrs(
    data_type="L1A_SCIDE>Level-1A Science Direct Events",
    logical_source="imap_lo_l1a_scide",
    logical_source_desc="IMAP Mission IMAP-Lo Instrument Level-1A Data",
    instrument_base=lo_base,
)

lo_l1b_global_attrs = GlobalDataLevelAttrs(
    data_type="L1B->Level-1B",
    logical_source="imap_lo_l1b",
    logical_source_desc="IMAP Mission IMAP-Lo Instrument Level-1B Data",
    instrument_base=lo_base,
)

lo_l1c_global_attrs = GlobalDataLevelAttrs(
    data_type="L1C->Level-1C",
    logical_source="imap_lo_l1c",
    logical_source_desc="IMAP Mission IMAP-Lo Instrument Level-1C Data",
    instrument_base=lo_base,
)

lo_tof_attrs = ScienceAttrs(
    validmin=0,
    validmax=GlobalConstants.INT_MAXVAL,
    catdesc="Time of Flight",
    # TODO: Not sure if depend_0 and display type are right
    depend_0="epoch",
    display_type="time_series",
    fill_val=GlobalConstants.INT_FILLVAL,
    fieldname="tof",
    format="I12",
    var_type="data",
    units="seconds",
    label_axis="Time of Flight",
)

"""IMAP-Lo L2 data processing."""

import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes


def lo_l2(sci_dependencies: dict, anc_dependencies: list) -> list[xr.Dataset]:
    """
    Will process IMAP-Lo L1C data into Le CDF data products.

    Parameters
    ----------
    sci_dependencies : dict
        Dictionary of datasets needed for L2 data product creation in xarray Datasets.
    anc_dependencies : list
        Ancillary files needed for L2 data product creation.

    Returns
    -------
    created_file_paths : list[Path]
        Location of created CDF files.
    """
    # create the attribute manager for this data level
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="lo", level="l1c")

    # if the dependencies are used to create Annotated Direct Events
    if "imap_lo_l1c_pset" in sci_dependencies:
        logical_source = "imap_lo_l2_hflux-spacecraft-uncorrected"
        hflux_spacecraft_uncorrected = xr.Dataset(
            attrs=attr_mgr.get_global_attributes(logical_source)
        )
        pset = sci_dependencies["imap_lo_l1c_pset"]

    return [hflux_spacecraft_uncorrected]
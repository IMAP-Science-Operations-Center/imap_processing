"""IMAP-Lo L2 data processing."""

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ena_maps import ena_maps
from imap_processing.spice import geometry


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
        # logical_source = "imap_lo_l2_hflux-spacecraft-uncorrected"
        pset = sci_dependencies["imap_lo_l1c_pset"]

        pset = pset.rename_dims(
            {"dim0": "longitude", "dim1": "latitude", "dim2": "energy"}
        )

        # Put energy dim before longitude and latitude
        for data_var in pset.data_vars:
            if "dim2" in pset[data_var].dims:
                # move dim2 to before dim0 and dim1
                pset[data_var] = pset[data_var].transpose(
                    "epoch", "energy", "longitude", "latitude"
                )

        pset["h_counts"] = xr.ones_like(pset["h_counts"])

        lo_rect_map = ena_maps.RectangularSkyMap(
            spacing_deg=6, spice_frame=geometry.SpiceFrame.ECLIPJ2000
        )

        lo_hp_map = ena_maps.HealpixSkyMap(
            nside=32,
            spice_frame=geometry.SpiceFrame.ECLIPJ2000,
        )

        lo_rect_map.project_pset_values_to_map(
            pointing_set=ena_maps.LoPointingSet(pset),
            value_keys=["h_counts", "exposure_time"],
            index_match_method=ena_maps.IndexMatchMethod.PUSH,
        )
        lo_hp_map.project_pset_values_to_map(
            pointing_set=ena_maps.LoPointingSet(pset),
            value_keys=["h_counts", "exposure_time"],
            index_match_method=ena_maps.IndexMatchMethod.PUSH,
        )

        lo_rect_map.data_1d["h_rate"] = (
            lo_rect_map.data_1d["h_counts"] / lo_rect_map.data_1d["exposure_time"]
        )
        # temporary values. These will all come from ancillary data when
        # the data is available.
        geometric_factor = 1.0
        efficiency_factor = 1.0
        energy_dict = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
        energies = np.array([energy_dict[i] for i in range(1, 8)])
        energies = energies.reshape(1, 7, 1)
        lo_rect_map.data_1d["h_flux"] = lo_rect_map.data_1d["h_rate"] / (
            geometric_factor * energies * efficiency_factor
        )

        lo_rect_map_ds = lo_rect_map.to_dataset()

    return [lo_rect_map_ds]

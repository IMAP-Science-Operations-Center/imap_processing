"""IMAP-Lo L2 data processing."""

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ena_maps import ena_maps
from imap_processing.ena_maps.ena_maps import RectangularSkyMap
from imap_processing.spice import geometry
from imap_processing.spice.geometry import SpiceFrame


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
    attr_mgr.add_instrument_variable_attrs(instrument="enamaps", level="l2-common")
    attr_mgr.add_instrument_variable_attrs(instrument="enamaps", level="l2-rectangular")

    # if the dependencies are used to create Annotated Direct Events
    if "imap_lo_l1c_pset" in sci_dependencies:
        logical_source = "imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-1yr"
        psets = sci_dependencies["imap_lo_l1c_pset"]

        # Create the rectangular sky map from the pointing set.
        lo_rect_map = project_pset_to_rect_map(
            psets, spacing_deg=6, spice_frame=geometry.SpiceFrame.ECLIPJ2000
        )
        # Add the hydrogen rates to the rectangular map dataset.
        lo_rect_map.data_1d["h_rate"] = calculate_rates(
            lo_rect_map.data_1d["h_counts"], lo_rect_map.data_1d["exposure_time"]
        )
        # Add the hydrogen flux to the rectangular map dataset.
        lo_rect_map.data_1d["h_flux"] = calculate_fluxes(lo_rect_map.data_1d["h_rate"])
        # Create the dataset from the rectangular map.
        lo_rect_map_ds = lo_rect_map.to_dataset()
        # Add the attributes to the dataset.
        # TODO: Temp quick fix for SIT-4. Pull into function and test after SIT-4.
        lo_rect_map_ds.attrs.update(attr_mgr.get_global_attributes(logical_source))

        # TODO: Lo is using different field names than what's in the attributes.
        #  check if the Lo should use exposure factor instead of exposure time.
        #  check if hydrogen and oxygen specific ena intensities should be added
        #  to the attributes or if general ena intensities can be used or updated
        #  in the code.
        lo_rect_map_ds.h_flux.attrs.update(
            attr_mgr.get_variable_attributes("ena_intensity")
        )
        lo_rect_map_ds.h_rate.attrs.update(attr_mgr.get_variable_attributes("ena_rate"))
        lo_rect_map_ds.h_counts.attrs.update(
            attr_mgr.get_variable_attributes("ena_count")
        )
        lo_rect_map_ds.exposure_time.attrs.update(
            attr_mgr.get_variable_attributes("exposure_factor")
        )

    return [lo_rect_map_ds]


def project_pset_to_rect_map(
    psets: list[xr.Dataset], spacing_deg: int, spice_frame: SpiceFrame
) -> RectangularSkyMap:
    """
    Project the pointing set to a rectangular sky map.

    This function is used to create a rectangular sky map from the pointing set
    data in the L1C dataset.

    Parameters
    ----------
    psets : list[xr.Dataset]
        List of pointing sets in xarray Dataset format.
    spacing_deg : int
        The spacing in degrees for the rectangular sky map.
    spice_frame : SpiceFrame
        The SPICE frame to use for the rectangular sky map projection.

    Returns
    -------
    RectangularSkyMap
        The rectangular sky map created from the pointing set data.
    """
    lo_rect_map = ena_maps.RectangularSkyMap(
        spacing_deg=spacing_deg,
        spice_frame=spice_frame,
    )
    for pset in psets:
        # Put energy dim before longitude and latitude
        # TODO: L1C data should be in this format already.
        #  This is a workaround for the current L1C data format.
        for data_var in pset.data_vars:
            if "energy" in pset[data_var].dims:
                # move dim2 to before dim0 and dim1
                pset[data_var] = pset[data_var].transpose(
                    "epoch", "energy", "longitude", "latitude"
                )

        lo_pset = ena_maps.LoPointingSet(pset)
        lo_rect_map.project_pset_values_to_map(
            pointing_set=lo_pset,
            value_keys=["h_counts", "exposure_time"],
            index_match_method=ena_maps.IndexMatchMethod.PUSH,
        )
    return lo_rect_map


def calculate_rates(counts: xr.DataArray, exposure_time: xr.DataArray) -> xr.DataArray:
    """
    Calculate the hydrogen rates from the counts and exposure time.

    Parameters
    ----------
    counts : xr.DataArray
        The counts of hydrogen or oxygen ENAs.
    exposure_time : xr.DataArray
        The exposure time for the counts.

    Returns
    -------
    xr.DataArray
        The calculated hydrogen rates.
    """
    # Calculate the rates based on the h_counts and exposure_time
    rate = counts / exposure_time
    return rate


def calculate_fluxes(rates: xr.DataArray) -> xr.DataArray:
    """
    Calculate the flux from the hydrogen rate.

    Parameters
    ----------
    rates : xr.Dataset
        The hydrogen or oxygen rates.

    Returns
    -------
    xr.DataArray
        The calculated flux.
    """
    # Temporary values. These will all come from ancillary data when
    # the data is available and integrated.
    geometric_factor = 1.0
    efficiency_factor = 1.0
    energy_dict = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    energies = np.array([energy_dict[i] for i in range(1, 8)])
    energies = energies.reshape(1, 7, 1)

    flux = rates / (geometric_factor * energies * efficiency_factor)
    return flux

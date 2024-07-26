"""Write Lo L1A CDFs."""

import numpy as np
import xarray as xr

from imap_processing.cdf import epoch_attrs
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import J2000_EPOCH
from imap_processing.lo.l0.lo_apid import LoAPID
from imap_processing.lo.l1a.lo_data_container import LoContainer


def write_lo_l1a_cdfs(data: LoContainer) -> list:
    """
    Write the Lo L1a CDFs.

    Parameters
    ----------
    data : LoContainer
        The Lo data container contaings all available Lo dataclass objects
        for one pointing.

    Returns
    -------
    created_file_paths : list[Path]
        Location of created CDF files.
    """
    created_datasets = []

    # Write Science Direct Events CDF if available
    science_direct_events = data.filter_apid(LoAPID.ILO_SCI_DE.value)
    if science_direct_events:
        scide_dataset = create_lo_scide_dataset(science_direct_events)
        created_datasets.append(scide_dataset)

    # TODO: Add the rest of the APIDS

    return created_datasets


def create_lo_scide_dataset(sci_de: list) -> xr.Dataset:
    """
    Create Lo L1A Science Direct Event Dataset from the ScienceDirectEvent dataclasses.

    Parameters
    ----------
    sci_de : list
        List of ScienceDirectEvent data classes.

    Returns
    -------
    sci_de_dataset : xarray.Dataset
        Lo L1A Science Direct Event Dataset.
    """
    # Load the CDF attributes
    cdf_manager = ImapCdfAttributes()
    cdf_manager.add_instrument_global_attrs("lo")
    cdf_manager.add_instrument_variable_attrs("lo", "l1a")
    # TODO: getting sci_de_times because it's used in both the data time field
    # and epoch. Need to figure out if this is needed and if a conversion needs
    # to happen to get the epoch time.
    sci_de_times = np.concatenate([sci_de_data.TIME for sci_de_data in sci_de])
    sci_de_time = xr.DataArray(
        sci_de_times, dims="epoch", attrs=cdf_manager.get_variable_attributes("de_time")
    )
    epoch_times = (
        np.array(sci_de_times, dtype="datetime64[s]").astype("datetime64[ns]")
        - J2000_EPOCH
    ).astype(np.int64)
    sci_de_epoch = xr.DataArray(
        epoch_times,
        dims=["epoch"],
        name="epoch",
        attrs=epoch_attrs,
    )
    sci_de_energy = xr.DataArray(
        np.concatenate([sci_de_data.ENERGY for sci_de_data in sci_de]),
        dims="epoch",
        attrs=cdf_manager.get_variable_attributes(
            "esa_step"
        ),  # TODO: check if this is correct
    )
    sci_de_mode = xr.DataArray(
        np.concatenate([sci_de_data.MODE for sci_de_data in sci_de]),
        dims="epoch",
        attrs=cdf_manager.get_variable_attributes("mode"),
    )
    sci_de_tof0 = xr.DataArray(
        np.concatenate([sci_de_data.TOF0 for sci_de_data in sci_de]),
        dims="epoch",
        attrs=cdf_manager.get_variable_attributes("tof0"),
    )
    sci_de_tof1 = xr.DataArray(
        np.concatenate([sci_de_data.TOF1 for sci_de_data in sci_de]),
        dims="epoch",
        attrs=cdf_manager.get_variable_attributes("tof1"),
    )
    sci_de_tof2 = xr.DataArray(
        np.concatenate([sci_de_data.TOF2 for sci_de_data in sci_de]),
        dims="epoch",
        attrs=cdf_manager.get_variable_attributes("tof2"),
    )
    sci_de_tof3 = xr.DataArray(
        np.concatenate([sci_de_data.TOF3 for sci_de_data in sci_de]),
        dims="epoch",
        attrs=cdf_manager.get_variable_attributes("tof3"),
    )
    sci_de_checksum = xr.DataArray(
        np.concatenate([sci_de_data.CKSM for sci_de_data in sci_de]),
        dims="epoch",
        attrs=cdf_manager.get_variable_attributes("cksm"),
    )
    sci_de_pos = xr.DataArray(
        np.concatenate([sci_de_data.POS for sci_de_data in sci_de]),
        dims="epoch",
        attrs=cdf_manager.get_variable_attributes("pos"),
    )

    # Create the full dataset
    sci_de_dataset = xr.Dataset(
        data_vars={
            "de_time": sci_de_time,
            "energy": sci_de_energy,
            "mode": sci_de_mode,
            "tof0": sci_de_tof0,
            "tof1": sci_de_tof1,
            "tof2": sci_de_tof2,
            "tof3": sci_de_tof3,
            "checksum": sci_de_checksum,
            "pos": sci_de_pos,
        },
        attrs=cdf_manager.get_global_attributes("imap_lo_l1a_de"),
        # TODO: figure out how to convert time data to epoch
        coords={"epoch": sci_de_epoch},
    )

    return sci_de_dataset

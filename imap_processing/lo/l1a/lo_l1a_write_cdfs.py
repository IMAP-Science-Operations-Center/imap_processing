"""Write Lo L1A CDFs."""

import numpy as np
import xarray as xr

from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import write_cdf
from imap_processing.lo.l0.lo_apid import LoAPID
from imap_processing.lo.l1a import lo_cdf_attrs
from imap_processing.lo.l1a.lo_data_container import LoContainer


# TODO: Add a main function that filters to Lo Level 0 data container object by APID
# and calls the appropriate l1a cdf writing function
def write_lo_l1a_cdfs(data: LoContainer):
    """
    Write the Lo L1a CDFs.

    Parameters
    ----------
    data : LoContainer
        The Lo data container contaings all available Lo dataclass objects.
    """
    science_direct_events = data.filter_apid(LoAPID.ILO_SCI_DE.value)
    if science_direct_events:
        scide_dataset = create_lo_scide_dataset(science_direct_events)
        write_cdf(scide_dataset)

    # TODO: Add the rest of the APIDS


def create_lo_scide_dataset(scide: list):
    """
    Create Lo L1A Science Direct Event Dataset from the ScienceDirectEvent dataclasses.

    Parameters
    ----------
    scide: list
        List of ScienceDirectEvent data classes.

    Returns
    -------
    xarray.Dataset
        Lo L1A Science Direct Event Dataset.
    """
    # Create each data array
    sci_de_checksum = xr.DataArray(
        np.concatenate([sci_de_data.CKSM for sci_de_data in scide]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_tof_attrs.output(),
    )
    sci_de_tof0 = xr.DataArray(
        np.concatenate([sci_de_data.TOF0 for sci_de_data in scide]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_tof_attrs.output(),
    )
    sci_de_tof1 = xr.DataArray(
        np.concatenate([sci_de_data.TOF1 for sci_de_data in scide]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_tof_attrs.output(),
    )
    sci_de_tof2 = xr.DataArray(
        np.concatenate([sci_de_data.TOF2 for sci_de_data in scide]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_tof_attrs.output(),
    )
    sci_de_tof3 = xr.DataArray(
        np.concatenate([sci_de_data.TOF3 for sci_de_data in scide]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_tof_attrs.output(),
    )
    sci_de_energy = xr.DataArray(
        np.concatenate([sci_de_data.ENERGY for sci_de_data in scide]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_tof_attrs.output(),
    )
    sci_de_pos = xr.DataArray(
        np.concatenate([sci_de_data.POS for sci_de_data in scide]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_tof_attrs.output(),
    )

    # TODO: getting sci_de_times because it's used in both the data time field
    # and epoch. Need to figure out if this is needed and if a conversion needs
    # to happen to get the epoch time.
    sci_de_times = np.concatenate([sci_de_data.TIME for sci_de_data in scide])

    sci_de_time = xr.DataArray(
        sci_de_times, dims="epoch", attrs=lo_cdf_attrs.lo_tof_attrs.output()
    )
    sci_de_epoch = xr.DataArray(
        np.array(sci_de_times, dtype="datetime64[s]"),
        dims=["epoch"],
        name="epoch",
        attrs=ConstantCoordinates.EPOCH,
    )

    # Create the full dataset
    sci_de_dataset = xr.Dataset(
        data_vars={
            "checksum": sci_de_checksum,
            "tof0": sci_de_tof0,
            "tof1": sci_de_tof1,
            "tof2": sci_de_tof2,
            "tof3": sci_de_tof3,
            "energy": sci_de_energy,
            "pos": sci_de_pos,
            "time": sci_de_time,
        },
        attrs=lo_cdf_attrs.lo_de_l1a_attrs.output(),
        # TODO: figure out how to convert time data to epoch
        coords={"epoch": sci_de_epoch},
    )

    return sci_de_dataset

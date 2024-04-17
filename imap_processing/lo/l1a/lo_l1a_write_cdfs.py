"""Write Lo L1A CDFs."""

import numpy as np
import xarray as xr

from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import write_cdf
from imap_processing.lo.l0.lo_apid import LoAPID
from imap_processing.lo.l1a import lo_cdf_attrs
from imap_processing.lo.l1a.lo_data_container import LoContainer


def write_lo_l1a_cdfs(data: LoContainer):
    """
    Write the Lo L1a CDFs.

    Parameters
    ----------
    data : LoContainer
        The Lo data container contaings all available Lo dataclass objects
        for one pointing.
    """
    created_filepaths = []

    # Write Science Direct Events CDF if available
    science_direct_events = data.filter_apid(LoAPID.ILO_SCI_DE.value)
    if science_direct_events:
        scide_dataset = create_lo_scide_dataset(science_direct_events)
        cdf_file = write_cdf(scide_dataset)
        created_filepaths.append(cdf_file)

    science_counts = data.filter_apid(LoAPID.ILO_SCI_CNT.value)
    # Write Science Counts CDF if available
    if science_counts:
        science_counts_dataset = create_lo_scicnt_dataset(science_counts)
        cdf_file = write_cdf(science_counts_dataset)
        created_filepaths.append(cdf_file)

    return created_filepaths


def create_lo_scide_dataset(sci_de: list):
    """
    Create Lo L1A Science Direct Event Dataset from the ScienceDirectEvent dataclasses.

    Parameters
    ----------
    sci_de: list
        List of ScienceDirectEvent data class objects.

    Returns
    -------
    xarray.Dataset
        Lo L1A Science Direct Event Dataset.
    """
    # TODO: getting sci_de_times because it's used in both the data time field
    # and epoch. Need to figure out if this is needed and if a conversion needs
    # to happen to get the epoch time.
    sci_de_times = np.concatenate([sci_de_data.TIME for sci_de_data in sci_de])
    sci_de_time = xr.DataArray(
        sci_de_times, dims="epoch", attrs=lo_cdf_attrs.lo_tof_attrs.output()
    )
    sci_de_epoch = xr.DataArray(
        np.array(sci_de_times, dtype="datetime64[s]"),
        dims=["epoch"],
        name="epoch",
        attrs=ConstantCoordinates.EPOCH,
    )
    sci_de_energy = xr.DataArray(
        np.concatenate([sci_de_data.ENERGY for sci_de_data in sci_de]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_tof_attrs.output(),
    )
    sci_de_mode = xr.DataArray(
        np.concatenate([sci_de_data.MODE for sci_de_data in sci_de]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_tof_attrs.output(),
    )
    sci_de_tof0 = xr.DataArray(
        np.concatenate([sci_de_data.TOF0 for sci_de_data in sci_de]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_tof_attrs.output(),
    )
    sci_de_tof1 = xr.DataArray(
        np.concatenate([sci_de_data.TOF1 for sci_de_data in sci_de]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_tof_attrs.output(),
    )
    sci_de_tof2 = xr.DataArray(
        np.concatenate([sci_de_data.TOF2 for sci_de_data in sci_de]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_tof_attrs.output(),
    )
    sci_de_tof3 = xr.DataArray(
        np.concatenate([sci_de_data.TOF3 for sci_de_data in sci_de]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_tof_attrs.output(),
    )
    sci_de_checksum = xr.DataArray(
        np.concatenate([sci_de_data.CKSM for sci_de_data in sci_de]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_tof_attrs.output(),
    )
    sci_de_pos = xr.DataArray(
        np.concatenate([sci_de_data.POS for sci_de_data in sci_de]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_tof_attrs.output(),
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
        attrs=lo_cdf_attrs.lo_de_l1a_attrs.output(),
        # TODO: figure out how to convert time data to epoch
        coords={"epoch": sci_de_epoch},
    )

    return sci_de_dataset


def create_lo_scicnt_dataset(sci_cnt: list):
    """
    Create Lo L1A Science Counts Dataset from the ScienceCounts dataclasses.

    Parameters
    ----------
    sci_cnt: list
        List of ScienceCounts data class objects.

    Returns
    -------
    xarray.Dataset
        Lo L1A Science Counts Dataset.
    """
    # TODO: How should the times be computed? Using START_A + SHCOARSE for now
    sci_cnt_times = np.concatenate(
        [sci_cnt_data.START_A + sci_cnt_data.SHCOARSE for sci_cnt_data in sci_cnt]
    )

    # TODO: Need to convert times epoch times to nanoseconds?
    sci_cnt_epoch = xr.DataArray(
        np.array(sci_cnt_times, dtype="datetime64[s]"),
        dims=["epoch"],
        name="epoch",
        attrs=ConstantCoordinates.EPOCH,
    )

    # create the dataset without populating data
    sci_cnt_dataset = xr.Dataset(
        data_vars={},
        attrs=lo_cdf_attrs.lo_sci_cnt_l1a_attrs.output(),
        # TODO: figure out how to convert time data to epoch
        coords={"epoch": sci_cnt_epoch},
    )

    # populate the dataset fields
    sci_cnt_dataset["start_a"] = xr.DataArray(
        np.concatenate([sci_cnt_data.START_A for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_anode_rate_attrs.output(),
    )
    sci_cnt_dataset["start_c"] = xr.DataArray(
        np.concatenate([sci_cnt_data.START_C for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_anode_rate_attrs.output(),
    )
    sci_cnt_dataset["stop_b0"] = xr.DataArray(
        np.concatenate([sci_cnt_data.STOP_B0 for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_ion_rate_attrs.output(),
    )
    sci_cnt_dataset["stop_b3"] = xr.DataArray(
        np.concatenate([sci_cnt_data.STOP_B3 for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_ion_rate_attrs.output(),
    )
    sci_cnt_dataset["tof0"] = xr.DataArray(
        np.concatenate([sci_cnt_data.TOF0 for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_tof_rate_attrs.output(),
    )
    sci_cnt_dataset["tof1"] = xr.DataArray(
        np.concatenate([sci_cnt_data.TOF1 for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_tof_rate_attrs.output(),
    )
    sci_cnt_dataset["tof2"] = xr.DataArray(
        np.concatenate([sci_cnt_data.TOF2 for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_tof_rate_attrs.output(),
    )
    sci_cnt_dataset["tof3"] = xr.DataArray(
        np.concatenate([sci_cnt_data.TOF3 for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_tof_rate_attrs.output(),
    )
    sci_cnt_dataset["tof0_tof1"] = xr.DataArray(
        np.concatenate([sci_cnt_data.TOF0_TOF1 for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_trip_coin_rate_attrs.output(),
    )
    sci_cnt_dataset["tof0_tof2"] = xr.DataArray(
        np.concatenate([sci_cnt_data.TOF0_TOF2 for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_trip_coin_rate_attrs.output(),
    )
    sci_cnt_dataset["tof1_tof2"] = xr.DataArray(
        np.concatenate([sci_cnt_data.TOF1_TOF2 for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_trip_coin_rate_attrs.output(),
    )
    sci_cnt_dataset["silver"] = xr.DataArray(
        np.concatenate([sci_cnt_data.SILVER for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_trip_coin_rate_attrs.output(),
    )
    sci_cnt_dataset["disc_tof0"] = xr.DataArray(
        np.concatenate([sci_cnt_data.DISC_TOF0 for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_disc_rate_attrs.output(),
    )
    sci_cnt_dataset["disc_tof1"] = xr.DataArray(
        np.concatenate([sci_cnt_data.DISC_TOF1 for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_disc_rate_attrs.output(),
    )
    sci_cnt_dataset["disc_tof2"] = xr.DataArray(
        np.concatenate([sci_cnt_data.DISC_TOF2 for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_disc_rate_attrs.output(),
    )
    sci_cnt_dataset["disc_tof3"] = xr.DataArray(
        np.concatenate([sci_cnt_data.DISC_TOF3 for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_disc_rate_attrs.output(),
    )
    sci_cnt_dataset["pos0"] = xr.DataArray(
        np.concatenate([sci_cnt_data.POS0 for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_pos_rate_attrs.output(),
    )
    sci_cnt_dataset["pos0"] = xr.DataArray(
        np.concatenate([sci_cnt_data.POS0 for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_pos_rate_attrs.output(),
    )
    sci_cnt_dataset["pos1"] = xr.DataArray(
        np.concatenate([sci_cnt_data.POS1 for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_pos_rate_attrs.output(),
    )
    sci_cnt_dataset["pos2"] = xr.DataArray(
        np.concatenate([sci_cnt_data.POS2 for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_pos_rate_attrs.output(),
    )
    sci_cnt_dataset["pos3"] = xr.DataArray(
        np.concatenate([sci_cnt_data.POS3 for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_pos_rate_attrs.output(),
    )
    sci_cnt_dataset["hydrogen"] = xr.DataArray(
        np.concatenate([sci_cnt_data.HYDROGEN for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_hydrogen_rate_attrs.output(),
    )
    sci_cnt_dataset["oxygen"] = xr.DataArray(
        np.concatenate([sci_cnt_data.OXYGEN for sci_cnt_data in sci_cnt]),
        dims="epoch",
        attrs=lo_cdf_attrs.lo_oxygen_rate_attrs.output(),
    )

    return sci_cnt_dataset

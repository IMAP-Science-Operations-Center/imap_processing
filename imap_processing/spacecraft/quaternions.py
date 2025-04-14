"""Spacecraft quaternion processing."""

from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.spice.time import met_to_ttj2000ns
from imap_processing.utils import packet_file_to_datasets


def load_quaternion_packets(packet_file: Path | str) -> xr.Dataset:
    """
    Load the raw quaternion packets from the packet file.

    Parameters
    ----------
    packet_file : Path
        Path to the packet file containing the quaternions in apid 594.

    Returns
    -------
    xarray.Dataset
        Dataset containing the raw quaternion packets.
    """
    xtce_packet_definition = Path(__file__).parent / "packet_definitions/scid_x252.xml"
    datasets_by_apid = packet_file_to_datasets(
        packet_file=packet_file, xtce_packet_definition=xtce_packet_definition
    )
    return datasets_by_apid[0x252]


def assemble_quaternions(ds: xr.Dataset) -> xr.Dataset:
    """
    Assemble quaternions from the l1a dataset.

    The quaternions are stored in separate variables for each component (x, y, z, s)
    and for each 10 Hz sample of the 1s packet. i.e. there are 4 * 10 = 40 variables
    in the initial dataset that we want to turn into 4 variables with a continuous
    10 Hz sampling period.

    The output dataset will have a single dimension "epoch" that will be the time
    associated with each of the 10 Hz samples. There are 4 data variables: "quat_x",
    "quat_y", "quat_z", "quat_s".

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing the 10Hz quaternions from packet 0x252 (APID 594).

    Returns
    -------
    xarray.Dataset
        Output dataset with the quaternions assembled into 4 variables with a
        continuous 10 Hz sampling period.
    """
    # Our time is only given for the first timestamp
    # We then add to it 0.1 increments and ravel the array to associate a specific time
    # with each of the 10 samples
    time = (
        ds["SCIENCEDATA1HZ_QUAT_10_HZ_TIME".lower()].values[:, np.newaxis]
        + np.arange(0, 1, 0.1)
    ).ravel()
    output_ds = xr.Dataset(coords={"epoch": time})
    base_name = "FSW_ACS_QUAT_10_HZ_BUFFERED".lower()
    for quat_i, label in enumerate(["x", "y", "z", "s"]):
        # 0, 1, 2, .. 9 // 10, 11, 12, .. 19 // 20, 21, 22, .. 29 // 30, 31, 32, .. 39
        names = [f"{base_name}_{i + quat_i * 10}" for i in range(10)]
        quat = np.stack([ds[name] for name in names], axis=1).ravel()
        output_ds[f"quat_{label}"] = ("epoch", quat)
    return output_ds


def process_quaternions(packet_file: Path | str) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Generate l1a and l1b datasets from a packet file containing the raw quaternions.

    This produces two CDF files: one for the l1a quaternions and one for the l1b
    quaternions. The l1b quaternions are assembled into 4 variables: "quat_x",
    "quat_y", "quat_z", "quat_s".

    Parameters
    ----------
    packet_file : Path
        Path to the packet file containing the quaternions in apid 594.

    Returns
    -------
    xarray.Dataset
        Dataset containing the l1a quaternions.
    xarray.Dataset
        Dataset containing the l1b quaternions.
    """
    l1a_ds = load_quaternion_packets(packet_file)

    # Assemble the quaternions into the correct components
    l1b_ds = assemble_quaternions(l1a_ds)

    # Update dataset global attributes
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs("spacecraft")
    attr_mgr.add_instrument_variable_attrs(instrument="spacecraft", level=None)

    l1a_ds.attrs.update(
        attr_mgr.get_global_attributes("imap_spacecraft_l1a_quaternions")
    )
    l1b_ds.attrs.update(
        attr_mgr.get_global_attributes("imap_spacecraft_l1b_quaternions")
    )

    # Update the epoch attribute
    l1b_ds["epoch"] = met_to_ttj2000ns(l1b_ds["epoch"])
    # check_schema=False keeps DEPEND_0 = '' from being auto added
    epoch_attrs = attr_mgr.get_variable_attributes("epoch", check_schema=False)
    l1b_ds["epoch"].attrs.update(epoch_attrs)

    for var in ["quat_x", "quat_y", "quat_z", "quat_s"]:
        l1b_ds[var].attrs.update(attr_mgr.get_variable_attributes(var))

    return l1a_ds, l1b_ds

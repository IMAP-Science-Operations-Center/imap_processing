"""
Perform IDEX L2b Processing.

Examples
--------
.. code-block:: python

    from imap_processing.idex.idex_l1a import PacketParser
    from imap_processing.idex.idex_l1b import idex_l1b
    from imap_processing.idex.idex_l1b import idex_l2a
    from imap_processing.idex.idex_l1b import idex_l2b

    l0_file = "imap_processing/tests/idex/imap_idex_l0_raw_20231218_v001.pkts"
    l0_file_hk = "imap_processing/tests/idex/imap_idex_l0_raw_20250108_v001.pkts"
    l1a_data = PacketParser(l0_file).data[0]
    evt_data = PacketParser(l0_file_hk).data[0]
    l1a_data, l1a_evt_data, l1b_evt_data = PacketParser(l0_file)
    l1b_data = idex_l1b(l1a_data)
    l1a_data = idex_l2a(l1b_data)
    l2b_data = idex_l2b(l2a_data, [evt_data])
    write_cdf(l2b_data)
"""

import logging

import numpy as np
import xarray as xr

from imap_processing.idex.idex_constants import IDEXEvtAcquireCodes
from imap_processing.idex.idex_utils import get_idex_attrs, setup_dataset
from imap_processing.spice.time import epoch_to_doy

logger = logging.getLogger(__name__)


def idex_l2b(l2a_dataset: xr.Dataset, evt_datasets: list[xr.Dataset]) -> xr.Dataset:
    """
    Will process IDEX l2a data to create l2b data products.

    Parameters
    ----------
    l2a_dataset : xarray.Dataset
        IDEX L2a dataset to process.
    evt_datasets : list[xarray.Dataset]
        List of IDEX housekeeping event message datasets.

    Returns
    -------
    l2b_dataset : xarray.Dataset
        The``xarray`` dataset containing the science data and supporting metadata.
    """
    logger.info(
        f"Running IDEX L2B processing on dataset: {l2a_dataset.attrs['Logical_source']}"
    )

    # create the attribute manager for this data level
    idex_attrs = get_idex_attrs("l2b")

    evt_dataset = xr.concat(evt_datasets, dim="epoch")

    # Create l2b Dataset
    prefixes = ["latitude", "longitude", "_dust_mass_estimate", "_impact_charge"]
    l2b_dataset = setup_dataset(l2a_dataset, prefixes, idex_attrs)
    l2b_dataset.attrs = idex_attrs.get_global_attributes("imap_idex_l2b_sci")

    # Get science acquisition start and stop times from event dataset
    evt_logs, evt_time, evt_values = get_science_acquisition_timestamps(evt_dataset)
    l2b_dataset["science_acquisition_messages"] = xr.DataArray(
        name="science_acquisition_messages",
        data=evt_logs.astype(str),
        dims="epoch_science_acquisition",
        attrs=idex_attrs.get_variable_attributes("science_acquisition_messages"),
    )
    l2b_dataset["epoch_science_acquisition"] = xr.DataArray(
        name="epoch_science_acquisition",
        data=evt_time,
        dims="epoch_science_acquisition",
        attrs=idex_attrs.get_variable_attributes(
            "epoch_science_acquisition", check_schema=False
        ),
    )
    l2b_dataset["science_acquisition_values"] = xr.DataArray(
        name="science_acquisition_values",
        data=evt_values,
        dims="epoch_science_acquisition",
        attrs=idex_attrs.get_variable_attributes("science_acquisition_values"),
    )
    spin_phase_quadrants = round_spin_phases(l2a_dataset["spin_phase"])
    spin_phase_quadrants.attrs.update(
        idex_attrs.get_variable_attributes("spin_phase_quadrants")
    )
    l2b_dataset["spin_phase_quadrants"] = spin_phase_quadrants

    # Get the time of impact array (in day of year)
    impact_day_of_year = epoch_to_doy(l2b_dataset["epoch"].data)
    l2b_dataset["impact_day_of_year"] = xr.DataArray(
        name="impact_day_of_year",
        data=impact_day_of_year,
        dims="epoch",
        attrs=idex_attrs.get_variable_attributes("impact_day_of_year"),
    )
    logger.info("IDEX L2B science data processing completed.")

    return l2b_dataset


def round_spin_phases(spin_phases: xr.DataArray) -> xr.DataArray:
    """
    Round spin phase angles to the nearest quadrant (0, 90, 180, 270).

    Parameters
    ----------
    spin_phases : xarray.DataArray
        Spacecraft spin phase angles. Expected to be integers in the range [0, 360).

    Returns
    -------
    xarray.DataArray
        Spin phases rounded to the nearest quadrant.
    """
    if np.any(spin_phases < 0) or np.any(spin_phases >= 360):
        logger.warning(
            f"Spin phase angles, {spin_phases.data} are outside of the expected spin "
            f"phase angle range, [0, 360)."
        )
    quadrant_size = 90
    # Shift spin phases so any value exactly between two quadrants gets shifted to the
    # Higher quadrant
    shifted_spin_phases = spin_phases + quadrant_size / 2
    # Calculate nearest quadrant value.
    # Use mod to wrap values > 315 to 0.
    return (quadrant_size * (shifted_spin_phases / quadrant_size).astype(int)) % 360


def get_science_acquisition_timestamps(
    evt_dataset: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the science acquisition start and stop times and messages from the event data.

    Parameters
    ----------
    evt_dataset : xarray.Dataset
        Contains IDEX event message data.

    Returns
    -------
    event_logs : np.ndarray
        Array containing science acquisition start and stop events messages.
    event_timestamps : np.ndarray
        Array containing science acquisition start and stop timestamps.
    event_values : np.ndarray
        Array containing values indicating if the event is a start (1) or
        stop (0).
    """
    # Sort the event dataset by the epoch time. Drop duplicates
    evt_dataset = evt_dataset.sortby("epoch").drop_duplicates("epoch")
    # First find indices of the state change events
    sc_indices = np.where(evt_dataset["elid_evtpkt"].data == "SCI_STE")[0]
    event_logs = []
    event_timestamps = []
    event_values = []
    # Get the values of the state change events
    val1 = (
        evt_dataset["el1par_evtpkt"].data[sc_indices] << 8
        | evt_dataset["el2par_evtpkt"].data[sc_indices]
    )
    val2 = (
        evt_dataset["el3par_evtpkt"].data[sc_indices] << 8
        | evt_dataset["el4par_evtpkt"].data[sc_indices]
    )
    epochs = evt_dataset["epoch"][sc_indices]
    # Now the state change values and check if it is either a science
    # acquisition start or science acquisition stop event.
    for v1, v2, epoch in zip(val1, val2, epochs):
        # An "acquire" start will have val1=ACQSETUP and val2=ACQ
        # An "acquire" stop will have val1=ACQ and val2=CHILL
        if (v1, v2) == (IDEXEvtAcquireCodes.ACQSETUP, IDEXEvtAcquireCodes.ACQ):
            event_logs.append("SCI state change: ACQSETUP to ACQ")
            event_timestamps.append(epoch)
            event_values.append(1)
        elif (v1, v2) == (IDEXEvtAcquireCodes.ACQ, IDEXEvtAcquireCodes.CHILL):
            event_logs.append("SCI state change: ACQ to CHILL")
            event_timestamps.append(epoch)
            event_values.append(0)

    return (
        np.asarray(event_logs),
        np.asarray(event_timestamps),
        np.asarray(event_values),
    )

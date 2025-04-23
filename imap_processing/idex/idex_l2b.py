"""
Perform IDEX L2b Processing.

Examples
--------
.. code-block:: python

    from imap_processing.idex.idex_l1a import PacketParser
    from imap_processing.idex.idex_l1b import idex_l1b
    from imap_processing.idex.idex_l1b import idex_l2a
    from imap_processing.idex.idex_l1b import idex_l2b

    l0_file = "imap_processing/tests/idex/imap_idex_l0_sci_20231214_v001.pkts"
    l1a_data = PacketParser(l0_file)
    l1b_data = idex_l1b(l1a_data)
    l1a_data = idex_l2a(l1b_data)
    l2b_data = idex_l2b(l2a_data)
    write_cdf(l2b_data)
"""

import logging

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.spice.time import epoch_to_doy

logger = logging.getLogger(__name__)


def idex_l2b(l2a_dataset: xr.Dataset) -> xr.Dataset:
    """
    Will process IDEX l2a data to create l2b data products.

    Parameters
    ----------
    l2a_dataset : xarray.Dataset
        IDEX L2a dataset to process.

    Returns
    -------
    l2b_dataset : xarray.Dataset
        The``xarray`` dataset containing the science data and supporting metadata.
    """
    logger.info(
        f"Running IDEX L2B processing on dataset: {l2a_dataset.attrs['Logical_source']}"
    )

    # create the attribute manager for this data level
    idex_attrs = ImapCdfAttributes()
    idex_attrs.add_instrument_global_attrs(instrument="idex")

    epoch_da = xr.DataArray(
        l2a_dataset["epoch"],
        name="epoch",
        dims=["epoch"],
        attrs=idex_attrs.get_variable_attributes("epoch"),
    )
    # Create l2b Dataset
    l2b_dataset = xr.Dataset(
        coords={"epoch": epoch_da},
        attrs=idex_attrs.get_global_attributes("imap_idex_l2b_sci"),
    )

    target_waveforms = ["target_high", "target_low", "ion_grid"]
    mass_name = "_fit_impact_mass_estimate"
    charge_name = "_fit_impact_charge"
    # Copy arrays to l2b dataset
    for waveform in target_waveforms:
        l2b_dataset[waveform + charge_name] = l2a_dataset[waveform + charge_name].copy(
            deep=True
        )
        l2b_dataset[waveform + mass_name] = l2a_dataset[waveform + mass_name].copy(
            deep=True
        )

    spin_phase_quadrants = round_spin_phases(l2a_dataset["spin_phase"])
    # TODO add variable attributes
    l2b_dataset["spin_phase_quadrants"] = spin_phase_quadrants

    # Get the time of impact array (in day of year)
    impact_day_of_year = epoch_to_doy(epoch_da.data)
    l2b_dataset["impact_day_of_year"] = xr.DataArray(
        name="impact_day_of_year",
        data=impact_day_of_year,
        dims="epoch",
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

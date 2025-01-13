"""
Perform IDEX L1b Processing.

Examples
--------
.. code-block:: python

    from imap_processing.idex.idex_l1a import PacketParser
    from imap_processing.idex.idex_l1b import idex_l1b

    l0_file = "imap_processing/tests/idex/imap_idex_l0_sci_20231214_v001.pkts"
    l1a_data = PacketParser(l0_file, data_version)
    l1b_data = idex_l1b(l1a_data, data_version)
    write_cdf(l1b_data)
"""

import logging
from enum import Enum
from typing import Union

import pandas as pd
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.utils import convert_raw_to_eu

logger = logging.getLogger(__name__)


class ConversionFactors(float, Enum):
    """Enum class for conversion factor values."""

    TOF_High = 2.89e-4
    TOF_Low = 5.14e-4
    TOF_Mid = 1.13e-2
    Target_Low = 1.58e1
    Target_High = 1.63e-1
    Ion_Grid = 7.46e-4


class TriggerMode(Enum):
    """Enum class for conversion factor values."""

    Threshold = 1
    SinglePulse = 2
    DoublePulse = 3

    @staticmethod
    def get_mode_label(mode: int, channel: str) -> str:
        """
        Return trigger mode label.

        Parameters
        ----------
        mode : int
            Raw mode value.
        channel : str
            Channel gain level.

        Returns
        -------
        str
            Mode label.
        """
        return f"{channel.upper()}{TriggerMode(mode).name}"


def idex_l1b(l1a_dataset: xr.Dataset, data_version: str) -> xr.Dataset:
    """
    Will process IDEX l1a data to create l1b data products.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        IDEX L1a dataset to process.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    l1b_dataset : xarray.Dataset
        The``xarray`` dataset containing the science data and supporting metadata.
    """
    logger.info(
        f"Running IDEX L1B processing on dataset: {l1a_dataset.attrs['Logical_source']}"
    )

    # create the attribute manager for this data level
    idex_attrs = ImapCdfAttributes()
    idex_attrs.add_instrument_global_attrs(instrument="idex")
    idex_attrs.add_instrument_variable_attrs(instrument="idex", level="l1b")
    idex_attrs.add_global_attribute("Data_version", data_version)

    var_information_path = (
        f"{imap_module_directory}/idex/idex_variable_unpacking_and_eu_conversion.csv"
    )
    # Read in csv that contains instrument variable setting information
    var_information_df = pd.read_csv(var_information_path)

    processed_vars = unpack_instrument_settings(
        l1a_dataset, var_information_df, idex_attrs
    )

    waveforms_converted = convert_waveforms(l1a_dataset, idex_attrs)

    epoch_da = xr.DataArray(
        l1a_dataset["epoch"],
        name="epoch",
        dims=["epoch"],
        attrs=idex_attrs.get_variable_attributes("epoch"),
    )

    trigger_settings = get_trigger_mode_and_level(l1a_dataset)
    if trigger_settings:
        trigger_settings["triggerlevel"].attrs = idex_attrs.get_variable_attributes(
            "trigger_level"
        )
        trigger_settings["triggermode"].attrs = idex_attrs.get_variable_attributes(
            "trigger_mode"
        )

    # Create l1b Dataset
    l1b_dataset = xr.Dataset(
        coords={"epoch": epoch_da},
        data_vars=processed_vars | waveforms_converted | trigger_settings,
        attrs=idex_attrs.get_global_attributes("imap_idex_l1b_sci"),
    )
    # Convert variables
    l1b_dataset = convert_raw_to_eu(
        l1b_dataset,
        conversion_table_path=var_information_path,
        packet_name="IDEX_SCI",
    )
    prefixes = ["shcoarse", "shfine", "time_high_sample", "time_low_sample"]
    vars_to_copy = [
        var
        for var in l1a_dataset.variables
        if any(prefix in var for prefix in prefixes)
    ]
    # Copy arrays from the l1a_dataset that do not need l1b processing
    for var in vars_to_copy:
        l1b_dataset[var] = l1a_dataset[var].copy()

    # TODO: Spice data?

    logger.info("IDEX L1B science data processing completed.")

    return l1b_dataset


def unpack_instrument_settings(
    l1a_dataset: xr.Dataset,
    var_information_df: pd.DataFrame,
    idex_attrs: ImapCdfAttributes,
) -> dict[str, xr.DataArray]:
    """
    Unpack raw telemetry data from the l1a dataset into individual variables.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        IDEX L1a dataset containing the 6 waveform arrays.
    var_information_df : pd.DataFrame
        Pandas data frame that contains information about each variable
        (e.g., bit-size, starting bit, and padding). This is used to unpack raw
        telemetry data from the input dataset (`l1a_dataset`).
    idex_attrs : ImapCdfAttributes
        CDF attribute manager object.

    Returns
    -------
    telemetry_data : dict
        A dictionary where the keys are the instrument setting array names and the
        values are the unpacked xr.DataArrays.
    """
    telemetry_data = {}

    for _, row in var_information_df.iterrows():
        unpacked_name = row["mnemonic"]

        # Create binary mask of the size of the variable in bits
        mask = (1 << row["unsigned_nbits"]) - 1
        # Determine the number of bits to shift
        shift = row["starting_bit"] - row["nbits_padding_before"]
        # Get the unpacked value by shifting the data to align the desired bits with
        # the least significant bits and applying the mask to isolate the target bits
        unpacked_val = (l1a_dataset[row["var_name"]].data >> shift) & mask

        telemetry_data[unpacked_name] = xr.DataArray(
            name=unpacked_name,
            data=unpacked_val,
            dims=("epoch"),
            attrs=idex_attrs.get_variable_attributes(unpacked_name),
        )

    return telemetry_data


def convert_waveforms(
    l1a_dataset: xr.Dataset, idex_attrs: ImapCdfAttributes
) -> dict[str, xr.DataArray]:
    """
    Apply transformation from raw DN to picocoulombs (pC) for each of the six waveforms.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        IDEX L1a dataset containing the six waveform arrays.
    idex_attrs : ImapCdfAttributes
        CDF attribute manager object.

    Returns
    -------
    waveforms_converted : dict
        A dictionary where the keys are the waveform array names and the values are
        xr.DataArrays representing the waveforms transformed into picocoulombs.
    """
    waveforms_pc = {}

    for var in ConversionFactors:
        waveforms_pc[var.name] = l1a_dataset[var.name] * var.value
        waveforms_pc[var.name].attrs = idex_attrs.get_variable_attributes(
            var.name.lower()
        )

    return waveforms_pc


def get_trigger_mode_and_level(
    l1a_dataset: xr.Dataset,
) -> Union[dict[str, xr.DataArray], dict]:
    """
    Determine the trigger mode and threshold level for each event.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        IDEX L1a dataset containing the six waveform arrays and instrument settings.

    Returns
    -------
    dict
        A dictionary containing the trigger mode and level values.
    """
    # low, mid, and high gain channels
    channels = ["lg", "mg", "hg"]
    # 10 bit mask
    mask = 0b1111111111
    trigger_modes = []
    trigger_levels = []

    def compute_trigger_values(
        trigger_mode: int, trigger_controls: int, gain_channel: str
    ) -> Union[tuple[str, Union[int, float]], tuple[None, None]]:
        """
        Compute the trigger mode label and threshold level.

        Parameters
        ----------
        trigger_mode : float
            Raw trigger mode value.
        trigger_controls : int
            Raw trigger control values.
        gain_channel : float
            Gain channel (low, mid, or high).

        Returns
        -------
        tuple
            Mode label and threshold level.
        """
        # If the trigger mode is zero, then the channel did not trigger the event and
        # therefore there is no threshold level
        if trigger_mode == 0:
            return None, None

        mode_label = TriggerMode.get_mode_label(mode=trigger_mode, channel=gain_channel)
        # The trigger control variable is 32 bits with the first 10 bits representing
        # the Threshold level.
        # Bit-shift right 22 places and use a 10-bit mask to extract the level value.
        threshold_level = float((trigger_controls >> 22) & mask)

        # If it is the high gain channel multiply the level by the conversion factor.
        # TODO: determine why the idex team is only doing this for the high gain channel
        if gain_channel == "hg":
            threshold_level *= ConversionFactors["TOF_High"]
        return mode_label, threshold_level

    for channel in channels:
        # Get all the modes and controls for each event for the current channel
        modes = l1a_dataset[f"idx__txhdr{channel}trigmode"].copy()
        controls = l1a_dataset[f"idx__txhdr{channel}trigctrl1"].copy()

        # Apply the function across the arrays
        mode_array, level_array = xr.apply_ufunc(
            compute_trigger_values,
            modes,
            controls,
            channel,
            output_core_dims=([], []),
            vectorize=True,
            output_dtypes=[object, float],
        )
        trigger_modes.append(mode_array.rename("trigger_mode"))
        trigger_levels.append(level_array.rename("trigger_level"))

    try:
        # There should be an array of modes and threshold levels for each channel.
        # At each index (event) only one of the three arrays should have a value that is
        # not 'None' because each event can only be triggered by one channel.
        # By merging the three arrays, we get value for each event.
        merged_modes = xr.merge([trigger_modes[0], xr.merge(trigger_modes[1:])])
        merged_levels = xr.merge([trigger_levels[0], xr.merge(trigger_levels[1:])])

        return {
            "triggermode": merged_modes.trigger_mode,
            "triggerlevel": merged_levels.trigger_level,
        }

    except xr.MergeError as e:
        raise ValueError(
            f"Only one channel can trigger a dust event. Please make sure "
            f"there is only one valid trigger value per event. This "
            f"caused Merge Error: {e}"
        ) from e

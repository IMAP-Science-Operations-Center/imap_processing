"""
Perform IDEX L1b Processing.

Examples
--------
.. code-block:: python

    from imap_processing.idex.idex_l1a import PacketParser
    from imap_processing.idex.idex_l1b import idex_l1b

    l0_file = "imap_processing/tests/idex/imap_idex_l0_sci_20231214_v001.pkts"
    l1a_data = PacketParser(l0_file)
    l1b_data = idex_l1b(l1a_data)
    write_cdf(l1b_data)
"""

import logging
from enum import Enum, IntEnum

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from xarray import DataArray

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.idex.idex_constants import (
    DT_BLOCK,
    IDEX_EVENT_REFERENCE_FRAME,
    ConversionFactors,
)
from imap_processing.idex.idex_utils import get_idex_attrs, setup_dataset
from imap_processing.spice.geometry import (
    SpiceBody,
    SpiceFrame,
    cartesian_to_spherical,
    imap_state,
    instrument_pointing,
    solar_longitude,
)
from imap_processing.spice.spin import get_spacecraft_spin_phase, get_spin_angle
from imap_processing.spice.time import et_to_met, ttj2000ns_to_et
from imap_processing.utils import convert_raw_to_eu

logger = logging.getLogger(__name__)


class EventMessage(Enum):
    """Enum class for event messages."""

    PULSER_ON = "SEQ success (len=0x0580, opCodeLCDictionary(enstim))"
    PULSER_OFF = "UPK stim pulser operation completed, , PulserSel=0x00000007"
    SCIENCE_ON = (
        "SCI state change: sciState16Dictionary(ACQSETUP) ==> sciState16Dictionary(ACQ)"
    )
    SCIENCE_OFF = (
        "SCI state change: sciState16Dictionary(ACQ) ==> sciState16Dictionary(CHILL)"
    )


class TriggerOrigin(IntEnum):
    """Enum class for event trigger origins."""

    HS_ADC0I_TOF_HG = 0
    HS_ADC0Q_TOF_LG = 1
    HS_ADC1Q_TOF_MG = 2
    LS_ADC1_TARGET_HG = 3
    SW_TRIGGER = 4
    EXTERNAL_TRIGGER = 5


TRIGGER_LABELS = {
    TriggerOrigin.HS_ADC0I_TOF_HG: "HS ADC0I trigger (TOF HG)",
    TriggerOrigin.HS_ADC0Q_TOF_LG: "HS ADC0Q trigger (TOF LG)",
    TriggerOrigin.HS_ADC1Q_TOF_MG: "HS ADC1Q trigger (TOF MG)",
    TriggerOrigin.LS_ADC1_TARGET_HG: "LS ADC1 trigger (Target HG / low range)",
    TriggerOrigin.SW_TRIGGER: "SW trigger",
    TriggerOrigin.EXTERNAL_TRIGGER: "external trigger",
}


class TriggerMode(Enum):
    """
    Enum class for data collection trigger Modes.

    Attributes
    ----------
    Threshold : int
        Mode 1 - Triggers when signal reaches the threshold value.
    SinglePulse : int
        Mode 2 - Triggers when a single pulse is detected.
    DoublePulse : int
        Mode 3 - Triggers when two pulses are detected.
    """

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


def idex_l1b(l1a_dataset: xr.Dataset, descriptor: str) -> xr.Dataset | None:
    """
    Process IDEX l1a data to create l1b data products based on the descriptor.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        IDEX L1a dataset to process.
    descriptor : str
        Descriptor to determine the type of l1b processing to perform. E.g. "sci-10days"
        or "msg-10days".

    Returns
    -------
    l1b_dataset : xarray.Dataset
        The``xarray`` dataset containing the processed data and supporting metadata.
    """
    if descriptor.startswith("sci-10days"):
        return idex_l1b_science(l1a_dataset)
    elif descriptor.startswith("msg-10days"):
        return idex_l1b_msg(l1a_dataset)
    else:
        raise ValueError(f"Unsupported descriptor: {descriptor}")


def idex_l1b_msg(l1a_dataset: xr.Dataset) -> xr.Dataset | None:
    """
    Will process IDEX l1a msg data.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        IDEX L1a dataset to process.

    Returns
    -------
    l1b_dataset : xarray.Dataset
        The``xarray`` dataset containing the msg housekeeping data and
        supporting metadata.
    """
    logger.info(
        f"Running IDEX L1B MSG processing on dataset: "
        f"{l1a_dataset.attrs['Logical_source']}"
    )
    # create the attribute manager for this data level
    idex_attrs = get_idex_attrs("l1b")
    # set up a dataset with only epoch.
    l1b_dataset = setup_dataset(l1a_dataset, [], idex_attrs, data_vars=None)
    l1b_dataset.attrs = idex_attrs.get_global_attributes("imap_idex_l1b_msg-10days")
    # Compute science_on and pulser_on variables based on the event message. The
    # "science_on" variable indicates when the science data collection is turned on or
    # off and the "pulser_on" variable indicates when the pulser is turned on or off.
    # The following logic is applied to determine the pulser_on status.
    # enstim → set pulser_on = 1
    # susprel AND the previous message was enstim → set pulser_on = 0
    # susprel but previous message was NOT enstim → pulser_on stays whatever it was
    l1a_messages = l1a_dataset.messages.values
    # Set science_on to 1 when science is on and 0 when it is off. 255 otherwise.
    science_on = np.where(l1a_messages == EventMessage.SCIENCE_ON.value, 1, 255)
    science_on[l1a_messages == EventMessage.SCIENCE_OFF.value] = 0
    # Find indices where there are consecutive PULSER_ON followed by PULSER_OFF
    # messages. These are the only cases where we should set pulser_on to 1 and 0.
    # Compare the messages by shifting the pulser off messages back by one and looking
    # for matching overlaps.
    consecutive_pulser_on_off = np.where(
        (l1a_messages[:-1] == EventMessage.PULSER_ON.value)
        & (l1a_messages[1:] == EventMessage.PULSER_OFF.value)
    )[0]
    pulser_on = np.full(len(l1a_messages), 255)  # initialize with 255 (unknown)
    pulser_on[consecutive_pulser_on_off] = 1
    pulser_on[consecutive_pulser_on_off + 1] = 0
    l1b_dataset["pulser_on"] = xr.DataArray(
        data=pulser_on,
        dims="epoch",
        name="pulser_on",
        attrs=idex_attrs.get_variable_attributes("pulser_on"),
    )
    l1b_dataset["science_on"] = xr.DataArray(
        data=science_on,
        dims="epoch",
        name="science_on",
        attrs=idex_attrs.get_variable_attributes("science_on"),
    )

    # Filter dataset to only include rows where there is an event
    # (either science or pulser)
    null_event = (pulser_on == 255) & (science_on == 255)
    l1b_dataset = l1b_dataset.isel(epoch=~null_event)
    if len(l1b_dataset["epoch"]) == 0:
        logger.warning(
            "No science or pulser events found. No l1b dataset will be created."
        )
        return None
    logger.info("IDEX L1B MSG data processing completed.")
    return l1b_dataset


def idex_l1b_science(l1a_dataset: xr.Dataset) -> xr.Dataset:
    """
    Will process IDEX l1a science data.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        IDEX L1a dataset to process.

    Returns
    -------
    l1b_dataset : xarray.Dataset
        The``xarray`` dataset containing the science data and supporting metadata.
    """
    logger.info(
        f"Running IDEX L1B processing on dataset: {l1a_dataset.attrs['Logical_source']}"
    )

    # create the attribute manager for this data level
    idex_attrs = get_idex_attrs("l1b")

    var_information_path = (
        f"{imap_module_directory}/idex/idex_variable_unpacking_and_eu_conversion.csv"
    )
    # Read in csv that contains instrument variable setting information
    var_information_df = pd.read_csv(var_information_path)

    processed_vars = unpack_instrument_settings(
        l1a_dataset, var_information_df, idex_attrs
    )

    dead_time = get_event_dead_time(l1a_dataset, idex_attrs)

    waveforms_converted = convert_waveforms(l1a_dataset, idex_attrs)

    # Get spice data and save them as xr.DataArrays in the output. Spice data is not
    # used for calculations yet but are saved in the CDF for reference.
    spice_data = get_spice_data(l1a_dataset, idex_attrs)

    trigger_settings = get_trigger_mode_and_level(l1a_dataset, idex_attrs)
    trigger_origin = get_trigger_origin(
        l1a_dataset["idx__txhdrtrigid"].data, idex_attrs
    )
    # Create l1b Dataset
    prefixes = ["shcoarse", "shfine", "time_high_sample", "time_low_sample", "aid"]
    data_vars = (
        processed_vars
        | dead_time
        | waveforms_converted
        | trigger_settings
        | spice_data
        | trigger_origin
    )
    l1b_dataset = setup_dataset(l1a_dataset, prefixes, idex_attrs, data_vars)
    l1b_dataset.attrs = idex_attrs.get_global_attributes("imap_idex_l1b_sci")
    # Convert variables
    l1b_dataset = convert_raw_to_eu(
        l1b_dataset,
        conversion_table_path=var_information_path,
        packet_name="IDEX_SCI",
    )

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
    var_information_df : pandas.DataFrame
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
    # Unpack each instrument setting only once (remove duplicated rows for segmented
    # polynomials)
    var_information_df = var_information_df.drop_duplicates(subset=["mnemonic"])
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
    Apply the channel-specific transformation from raw DN to engineering units.

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
        xr.DataArrays representing the converted waveforms. TOF channels are in mA;
        target and ion-grid channels are in pC.
    """
    waveforms_converted = {}

    for var in ConversionFactors:
        waveforms_converted[var.name] = l1a_dataset[var.name] * var.value
        waveforms_converted[var.name].attrs = idex_attrs.get_variable_attributes(
            var.name.lower()
        )

    return waveforms_converted


def get_trigger_mode_and_level(
    l1a_dataset: xr.Dataset,
    idex_attrs: ImapCdfAttributes,
) -> dict[str, xr.DataArray] | dict:
    """
    Determine the trigger mode and threshold level for each event.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        IDEX L1a dataset containing the six waveform arrays and instrument settings.
    idex_attrs : ImapCdfAttributes
        CDF attribute manager object.

    Returns
    -------
    dict
        A dictionary containing the trigger mode and level values.
    """
    # low, mid, and high gain channels
    channels = ["lg", "mg", "hg"]
    # 10 bit mask
    mask = 0b1111111111
    # Initialize a dict to hold the mode labels and threshold levels for each channel
    data_dict = {}

    def compute_trigger_values(
        trigger_mode: int, trigger_controls: int, gain_channel: str
    ) -> tuple[str, int | float] | tuple[None, None]:
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

        # multiply the threshold level by the conversion factor.
        if gain_channel == "hg":
            threshold_level *= ConversionFactors["TOF_High"]
        elif gain_channel == "mg":
            threshold_level *= ConversionFactors["TOF_Mid"]
        elif gain_channel == "lg":
            threshold_level *= ConversionFactors["TOF_Low"]

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
        # Allocate the object array explicitly.  Otherwise pandas 3 string
        # inference converts the no-trigger None values to NaN.
        mode_values = np.asarray(mode_array.data, dtype=object)
        mode_values[pd.isna(mode_values)] = None
        object_mode_array = xr.full_like(modes, None, dtype=object)
        object_mode_array.data[:] = mode_values
        mode_array = object_mode_array
        # There should be an array of modes and threshold levels for each channel.
        # write each of them out as separate variables because there may be
        # multiple channels that can trigger an event. The trigger origin variable
        # can be used to determine which channel(s) triggered the event.
        mode_array.attrs = idex_attrs.get_variable_attributes(f"trigger_mode_{channel}")
        data_dict[f"trigger_mode_{channel}"] = mode_array
        level_array.attrs = idex_attrs.get_variable_attributes(
            f"trigger_level_{channel}"
        )
        data_dict[f"trigger_level_{channel}"] = level_array

    return data_dict


def get_trigger_origin(
    trigger_id: NDArray, idex_attrs: ImapCdfAttributes
) -> dict[str, DataArray]:
    """
    Determine the trigger origin for each event.

    Parameters
    ----------
    trigger_id : numpy.ndarray
        Array of raw trigger ID values from the l1a dataset. The trigger ID is a 32-bit
        integer where the lower 10 bits contain information about the trigger origin.
    idex_attrs : ImapCdfAttributes
        CDF attribute manager object.

    Returns
    -------
    dict[str, xarray.DataArray]
        A dictionary containing the trigger_origin DataArray with the trigger
        origin info for each event.
    """
    # extract the lower 10 bits of the trigger ID to get the trigger origin information
    trigger_bits = trigger_id & 0x3FF
    # For each event, determine which bits are set and get the corresponding trigger
    # origin labels
    origin_labels = np.array(
        [
            ", ".join(
                [TRIGGER_LABELS[TriggerOrigin(i)] for i in range(6) if (bits >> i) & 1]
            )
            for bits in trigger_bits
        ],
        dtype=object,
    )
    # Update any events with no trigger bits set to "unknown trigger origin"
    origin_labels[origin_labels == ""] = "Unknown trigger origin"
    return {
        "trigger_origin": xr.DataArray(
            name="trigger_origin",
            data=np.squeeze(origin_labels),
            dims="epoch",
            attrs=idex_attrs.get_variable_attributes("trigger_origin"),
        )
    }


def get_event_dead_time(
    l1a_dataset: xr.Dataset,
    idex_attrs: ImapCdfAttributes,
) -> dict[str, xr.DataArray]:
    """
    Compute event dead time (in seconds) from packed txhdrblocks.

    The dead time is encoded via two bitfields:
    - dead_blocks_base  (6 bits)
    - dead_blocks_shift (4 bits)

    Dead time is computed as:
        dead_time = dead_blocks_base * 2**dead_blocks_shift * DT_BLOCK

    where DT_BLOCK is the duration of a single low-rate block.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        IDEX L1A dataset containing the packed `idx__txhdrblocks` variable.
    idex_attrs : ImapCdfAttributes
        CDF attribute manager object.

    Returns
    -------
    dict[str, xarray.DataArray]
        Dictionary containing the `dead_time` DataArray (seconds).
    """
    txhdrblocks = l1a_dataset["idx__txhdrblocks"].data

    # Extract bitfields
    dead_blocks_shift = (txhdrblocks >> 20) & 0b1111
    dead_blocks_base = (txhdrblocks >> 24) & 0b111111

    # Convert to float once
    base = dead_blocks_base.astype(np.float64)
    shift = dead_blocks_shift.astype(np.float64)

    # Compute dead time
    dead_time_array: NDArray[np.float64] = base * np.power(2.0, shift) * DT_BLOCK

    return {
        "dead_time": xr.DataArray(
            name="dead_time",
            data=dead_time_array,
            dims="epoch",
            attrs=idex_attrs.get_variable_attributes("dead_time"),
        )
    }


def get_spice_data(
    l1a_dataset: xr.Dataset, idex_attrs: ImapCdfAttributes
) -> dict[str, xr.DataArray]:
    """
    Use spice to query ephemeris, attitude, celestial coordinates for each dust event.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        IDEX L1a dataset containing the six waveform arrays and instrument settings.
    idex_attrs : ImapCdfAttributes
        CDF attribute manager object.

    Returns
    -------
    dict
        Spice array names and xr.DataArrays.
    """
    # convert 'epoch' from nanoseconds to seconds since j2000
    et = ttj2000ns_to_et(l1a_dataset["epoch"].data)
    # Get (Mission Elapsed Time)
    met = et_to_met(et)
    # Get spacecraft spin phase in degrees
    spin_phase = get_spacecraft_spin_phase(query_met_times=met)
    imap_spin_phase = get_spin_angle(spin_phase, degrees=True)
    # Get the position and velocity of IMAP in ecliptic frame
    ephemeris = imap_state(et, observer=SpiceBody.SUN)
    # Get Idex pointing in the defined frame
    idex_pointing = instrument_pointing(
        et, SpiceFrame.IMAP_IDEX, IDEX_EVENT_REFERENCE_FRAME, cartesian=True
    )
    solar_lon = solar_longitude(et, degrees=True)
    # longitude and latitude
    lon_and_lat = cartesian_to_spherical(idex_pointing)[:, 1:]

    spice_data = {
        "ephemeris_position_x": ephemeris[:, 0],
        "ephemeris_position_y": ephemeris[:, 1],
        "ephemeris_position_z": ephemeris[:, 2],
        "ephemeris_velocity_x": ephemeris[:, 3],
        "ephemeris_velocity_y": ephemeris[:, 4],
        "ephemeris_velocity_z": ephemeris[:, 5],
        "longitude": lon_and_lat[:, 0],
        "latitude": lon_and_lat[:, 1],
        "spin_phase": imap_spin_phase,
        "solar_longitude": solar_lon,
    }

    for name, array in spice_data.items():
        spice_data[name] = xr.DataArray(
            name=name,
            data=array,
            dims="epoch",
            attrs=idex_attrs.get_variable_attributes(name),
        )

    return spice_data

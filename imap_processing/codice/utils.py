"""
Various classes and functions used throughout CoDICE processing.

This module contains utility classes and functions that are used by various
other CoDICE processing modules.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.codice import constants
from imap_processing.codice.constants import CODICEAPID


@dataclass
class ViewTabInfo:
    """
    Class to hold view table information.

    Attributes
    ----------
    apid : int
        The APID for the packet.
    collapse_table : int
        Collapse table id used to determine the collapse pattern.
    compression : int
        Compression algorithm used for the packet.
    sensor : int
        Sensor id (0 for LO, 1 for HI).
    three_d_collapsed : int
        The 3D collapsed value from the LUT.
    view_id : int
        The view identifier from the packet.
    """

    apid: int
    collapse_table: int
    compression: int
    sensor: int
    three_d_collapsed: int
    view_id: int


class CoDICECompression(IntEnum):
    """Create ENUM for CoDICE compression algorithms."""

    NO_COMPRESSION = 0
    LOSSY_A = 1
    LOSSY_B = 2
    LOSSLESS = 3
    LOSSY_A_LOSSLESS = 4
    LOSSY_B_LOSSLESS = 5
    PACK_24_BIT = 6


class SegmentedPacketOrder(IntEnum):
    """ENUM for segmented packet order."""

    UNSEGMENTED = 3
    FIRST_SEGMENT = 1
    CONTINUATION_SEGMENT = 0
    LAST_SEGMENT = 2


def read_sci_lut(file_path: Path, table_id: str) -> dict:
    """
    Read the SCI-LUT JSON file for a specific table ID.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the SCI-LUT JSON file.
    table_id : str
        Table identifier to extract from the JSON.

    Returns
    -------
    dict
        The SCI-LUT data for the specified table id.
    """
    sci_lut_data = json.loads(file_path.read_text()).get(f"{table_id}")
    if sci_lut_data is None:
        raise ValueError(f"SCI-LUT file does not have data for table ID {table_id}.")
    return sci_lut_data


def get_view_tab_info(json_data: dict, view_id: int, apid: int) -> dict:
    """
    Get the view table information for a specific view and APID.

    Parameters
    ----------
    json_data : dict
        The JSON data loaded from the SCI-LUT file.
    view_id : int
        The view ID from the packet.
    apid : int
        The APID from the packet.

    Returns
    -------
    dict
        The view table information containing details like sensor,
        collapse_table, data_product, etc.
    """
    apid_hex = f"0x{apid:X}"
    # This is how we get view information that will be used to get
    # collapse pattern:
    #  table_id -> view_tab -> (view_id, apid) -> sensor -> collapse_table ->compression
    # 'view_tab': {'(0, 0x480)': {'collapse_table': 0, '3d_collapse': 1, 'sensor': 0,
    # 'compression':0}
    view_tab = json_data.get("view_tab").get(f"({view_id}, {apid_hex})")
    return view_tab


def get_view_tab_obj(
    lut_file: Path, table_id: str, view_id: int, apid: int
) -> tuple[dict, "ViewTabInfo"]:
    """
    Read the SCI-LUT and build a ViewTabInfo for the given table ID.

    Parameters
    ----------
    lut_file : pathlib.Path
        Path to the SCI-LUT JSON file.
    table_id : str
        Table identifier to extract from the JSON.
    view_id : int
        The view ID from the packet.
    apid : int
        The APID from the packet.

    Returns
    -------
    tuple[dict, ViewTabInfo]
        The SCI-LUT data dict and a populated ViewTabInfo for the given table ID.
    """
    sci_lut_data = read_sci_lut(lut_file, table_id)
    view_tab_info = get_view_tab_info(sci_lut_data, view_id, apid)
    view_tab_obj = ViewTabInfo(
        apid=apid,
        view_id=view_id,
        sensor=view_tab_info["sensor"],
        three_d_collapsed=view_tab_info["3d_collapse"],
        collapse_table=view_tab_info["collapse_table"],
        compression=view_tab_info["compression"],
    )
    return sci_lut_data, view_tab_obj


def process_by_table_id(
    unpacked_dataset: xr.Dataset,
    lut_file: Path,
    process_fn: Callable[..., xr.Dataset],
) -> xr.Dataset:
    """
    Split dataset by unique table_id values, process each group, and recombine.

    This is the shared wrapper logic used by all non-DE/NHK L1A processing
    functions. It extracts the fields that are uniform across a packet stream
    (``view_id``, ``apid``, ``plan_id``, ``plan_step``), iterates over every
    unique ``table_id`` found in the dataset, filters to that group via
    ``isel``, calls *process_fn* for each group, and finally concatenates the
    results sorted by epoch.

    Parameters
    ----------
    unpacked_dataset : xarray.Dataset
        Full unpacked dataset from the L0 packet file.
    lut_file : pathlib.Path
        Path to the SCI-LUT JSON file passed through to *process_fn*.
    process_fn : Callable
        The private ``_process_xxx`` function to call for each table_id group.
        It must accept the signature
        ``(group_ds, lut_file, table_id, view_id, apid, plan_id, plan_step)``
        and return an ``xr.Dataset``.

    Returns
    -------
    xarray.Dataset
        Combined L1A dataset sorted by epoch.
    """
    view_id = unpacked_dataset["view_id"].values[0]
    apid = unpacked_dataset["pkt_apid"].values[0]
    plan_id = unpacked_dataset["plan_id"].values[0]
    plan_step = unpacked_dataset["plan_step"].values[0]

    unique_table_ids = np.unique(unpacked_dataset["table_id"].values)
    processed = [
        process_fn(
            unpacked_dataset.isel(
                epoch=unpacked_dataset["table_id"].values == table_id
            ),
            lut_file,
            table_id,
            view_id,
            apid,
            plan_id,
            plan_step,
        )
        for table_id in unique_table_ids
    ]
    if len(processed) == 1:
        return processed[0]
    # Keep non-epoch support variables as it is, 1-D arrays,
    # instead of expanding them along the epoch dimension.
    # Eg. voltage_table and k_factor are 1-D arrays that apply to all
    # epochs in the same way.
    return xr.concat(
        processed,
        dim="epoch",
        data_vars="minimal",
        coords="minimal",
        compat="equals",
    ).sortby("epoch")


def get_collapse_pattern_shape(
    json_data: dict, sensor_id: int, collapse_table_id: int
) -> tuple[int, ...]:
    """
    Get the collapse pattern for a specific sensor id and collapse table id.

    Parameters
    ----------
    json_data : dict
        The JSON data loaded from the SCI-LUT file.
    sensor_id : int
        Sensor identifier (0 for LO, 1 for HI).
    collapse_table_id : int
        Collapse table id to look up in the SCI-LUT.

    Returns
    -------
    tuple[int, int]
        (<spin_sector, inst_azimuth>) describing the collapsed pattern. Examples:
        ``(1,)`` for a fully collapsed 1-D pattern or ``(N, M)`` for a
        reduced 2-D pattern.
    """
    sensor = "lo" if sensor_id == 0 else "hi"
    collapse_matrix = np.array(
        json_data[f"collapse_{sensor}"][f"{collapse_table_id}"]["matrix"]
    )

    # Analyze the collapse pattern matrix to determine its reduced shape.
    # Steps:
    # - Extract non-zero elements from the matrix.
    # - Reshape to group unique non-zero rows and columns.
    # - If all non-zero values are identical, return (1,) for a fully collapsed pattern.
    # - Otherwise, compute the number of unique rows and columns to describe the
    #   reduced shape.
    non_zero_data = np.where(collapse_matrix != 0)
    non_zero_reformatted = collapse_matrix[non_zero_data].reshape(
        np.unique(non_zero_data[0]).size, np.unique(non_zero_data[1]).size
    )

    if np.unique(non_zero_reformatted).size == 1:
        # all non-zero values are identical means -> fully collapsed
        return (1,)

    # If not fully collapsed, find repeated patterns in rows and columns
    # to reduce shape further.
    unique_rows = np.unique(non_zero_reformatted, axis=0)
    unique_columns = np.unique(non_zero_reformatted, axis=1)
    # Unique spin sectors and instrument azimuths to unpack data
    unique_spin_sectors = unique_columns.shape[1]
    unique_inst_azs = unique_rows.shape[0]
    return (unique_spin_sectors, unique_inst_azs)


def get_counters_aggregated_pattern(
    json_data: dict, sensor_id: int, collapse_table_id: int
) -> dict:
    """
    Return the aggregated counters pattern from the SCI-LUT JSON.

    The counters aggregated pattern is stored as {key: list} in the SCI-LUT JSON.
    Each variable can be turned on and off in-flight. Because of that, we need to
    be flexible. If any variable is turned off, its corresponding row in the
    matrix will be all zeros and fill CDF variable for that row with zeros.

    Parameters
    ----------
    json_data : dict
        The JSON data loaded from the SCI-LUT file.
    sensor_id : int
        Sensor identifier (0 for LO, 1 for HI).
    collapse_table_id : int
        Collapse table id to look up in the SCI-LUT.

    Returns
    -------
    dict
        The counters key and its corresponding collapse pattern.
    """
    sensor = "lo" if sensor_id == 0 else "hi"
    full_matrix = json_data[f"collapse_{sensor}"][f"{collapse_table_id}"]["variables"]
    # Filter non-zero rows only
    non_zero_rows = {
        k: data_list for k, data_list in full_matrix.items() if 0 not in data_list
    }
    # Sort keys in order of unique num of their list.
    #   Eg. CoDICE Hi's counters-aggregated is not collapsed
    #   in the order of row by row. It could have collected in this order:
    #   [
    #       [1....1],
    #       [2....2],
    #       [3....3],
    #       [4....4],
    #       [7....7],
    #       [8....8],
    #       [11....11],
    #       [5....5],
    #       [6....6],
    #       [9....9],
    #       [10....10],
    #   ]
    #   Sort to get:
    #   [
    #       [1....1],
    #       [2....2],
    #       ...
    #       [11....11],
    in_order_rows = dict(sorted(non_zero_rows.items(), key=lambda item: item[1][0]))
    # Now get collapse pattern for all variables by finding
    # collapse pattern for the first key. Then replace all key's
    # with that because it should be same. If not,
    # that will effect these remaining logic.
    first_key = next(iter(in_order_rows))
    collapse_patterns = np.array(in_order_rows[first_key])
    # We only look for collapse pattern of columns because each variable
    # are rows in the collapse pattern matrix.
    unique_columns = np.unique(collapse_patterns, axis=0)
    unique_spin_sectors = unique_columns.shape[0]
    for key in in_order_rows:
        in_order_rows[key] = unique_spin_sectors
    return in_order_rows


def index_to_position(
    json_data: dict, sensor_id: int, collapse_table_id: int
) -> np.ndarray:
    """
    Get the indices of non-zero unique rows in the collapse pattern matrix.

    Parameters
    ----------
    json_data : dict
        The JSON data loaded from the SCI-LUT file.
    sensor_id : int
        Sensor identifier (0 for LO, 1 for HI).
    collapse_table_id : int
        Collapse table id to look up in the SCI-LUT.

    Returns
    -------
    np.ndarray
        Array of indices corresponding to non-zero unique rows.
    """
    sensor = "lo" if sensor_id == 0 else "hi"
    collapse_matrix = np.array(
        json_data[f"collapse_{sensor}"][f"{collapse_table_id}"]["matrix"]
    )

    # Find unique non-zero rows and their original indices
    non_zero_row_mask = np.any(collapse_matrix != 0, axis=1)
    non_zero_rows = collapse_matrix[non_zero_row_mask]
    _, unique_indices = np.unique(non_zero_rows, axis=0, return_index=True)
    non_zero_row_indices = np.flatnonzero(non_zero_row_mask)[unique_indices]
    return non_zero_row_indices


def get_codice_epoch_time(
    acq_start_seconds: np.ndarray,
    acq_start_subseconds: np.ndarray,
    spin_period: np.ndarray,
    view_tab_obj: ViewTabInfo,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate center time and delta.

    Parameters
    ----------
    acq_start_seconds : np.ndarray
        Array of acquisition start seconds.
    acq_start_subseconds : np.ndarray
        Array of acquisition start subseconds.
    spin_period : np.ndarray
        Array of spin periods.
    view_tab_obj : ViewTabInfo
        The view table information object. It contains information such as sensor ID
        and three_d_collapsed value and others.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (center_times (s), delta_times (ns)). center_times is converted to
        nanoseconds at CDF write time. delta_times is returned as integer
        nanoseconds.
    """
    # If Lo sensor
    if view_tab_obj.sensor == 0:
        # Lo sensor, we need to set spins to be constant.
        # 32 half spins makes full 16 spins for all non direct event products.
        # But Lo direct event's spins is also 16 spins. Because of that, we can use
        # the same calculation for all Lo products.
        num_spins = 16
    # If Hi sensor and Direct Event product
    elif view_tab_obj.sensor == 1 and view_tab_obj.apid == CODICEAPID.COD_HI_PHA:
        # Use constant 16 spins for Hi PHA
        num_spins = 16
    # If Non-Direct Event Hi product
    else:
        # Use 3d_collapsed value from LUT for other Hi products
        num_spins = int(view_tab_obj.three_d_collapsed)

    # Units of 'spin ticks', where one 'spin tick' equals 320 microseconds.
    # It takes multiple spins to collect data for a view. Keep the full delta
    # calculation in integer nanoseconds so the written CDF type is CDF_INT8.
    spin_period_ns: np.ndarray = spin_period.astype(np.int64) * 320_000
    delta_times: np.ndarray = (num_spins * spin_period_ns) // 2
    # subseconds need to converted to seconds using this formula per CoDICE team:
    #   subseconds / 65536 gives seconds
    center_times_seconds = (
        acq_start_seconds + acq_start_subseconds / 65536 + (delta_times / 1e9)
    )

    return center_times_seconds, delta_times


def calculate_acq_time_per_step(
    low_stepping_tab: dict, esa_step_dim: int = 128
) -> np.ndarray:
    """
    Calculate acquisition time per step from low stepping table.

    Parameters
    ----------
    low_stepping_tab : dict
        The low stepping table from the SCI-LUT JSON.
    esa_step_dim : int
        The ESA step dimension size.

    Returns
    -------
    np.ndarray
        Array of acquisition times per step of shape (num_esa_steps,).
    """
    # These tunable values are used to calculate acquisition time per step
    tunable_values = low_stepping_tab["tunable_values"]

    # pre-calculate values
    sector_time = tunable_values["spin_time_ms"] / tunable_values["num_sectors_ms"]
    sector_margin_ms = tunable_values["sector_margin_ms"]
    dwell_fraction = tunable_values["dwell_fraction_percentage"]
    min_hv_settle_ms = tunable_values["min_hv_settle_ms"]
    max_hv_settle_ms = tunable_values["max_hv_settle_ms"]
    num_steps_data = np.array(
        low_stepping_tab["num_steps"].get("data"), dtype=np.float64
    )
    # If num_steps_data is less than 128, pad with nan
    if len(num_steps_data) < constants.NUM_ESA_STEPS:
        pad_size = esa_step_dim - len(num_steps_data)
        num_steps_data = np.concatenate((num_steps_data, np.full(pad_size, np.nan)))
    # Total non-acquisition time is in column (BD) of science LUT
    dwell_fraction_percentage = float(sector_time) * (100.0 - dwell_fraction) / 100.0

    # Calculate HV settle time per step not adjusted for Min/Max.
    # It's in column (BF) of science LUT.
    non_adjusted_hv_settle_per_step = (
        dwell_fraction_percentage - sector_margin_ms
    ) / num_steps_data
    hv_settle_per_step = np.minimum(
        np.maximum(non_adjusted_hv_settle_per_step, min_hv_settle_ms), max_hv_settle_ms
    )
    # initialize array of nans for acquisition time per step
    acq_time_per_step: np.ndarray = np.full(esa_step_dim, np.nan, dtype=np.float64)
    # acquisition time per step in milliseconds
    # sector_time - sector_margin_ms / num_steps - hv_settle_per_step
    acq_time_per_step[: len(num_steps_data)] = (
        (sector_time - sector_margin_ms) / num_steps_data
    ) - hv_settle_per_step
    # Convert to seconds
    return acq_time_per_step / 1e3


def get_energy_info(
    energy_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate energy bin centers and deltas from energy table.

    Parameters
    ----------
    energy_table : np.ndarray
        The species plus and minus energy array.

    Returns
    -------
    centers : np.ndarray
        The geometric centers of the energy bins.
    deltas_minus : np.ndarray
        The delta minus values of the energy bins.
    deltas_plus : np.ndarray
        The delta plus values of the energy bins.
    """
    # Find the geometric centers and deltas of the energy bins
    # The delta minus is the difference between the center of the bin
    # and the 'left edge' of the bin. The delta plus is the difference
    # between the 'right edge' of the bin and the center of the bin
    min_energy = np.array(energy_table["min_energy"], dtype=np.float64)
    max_energy = np.array(energy_table["max_energy"], dtype=np.float64)

    centers = np.sqrt(min_energy * max_energy)
    deltas_minus = centers - min_energy
    deltas_plus = max_energy - centers

    return centers, deltas_minus, deltas_plus


def apply_replacements_to_attrs(attrs: dict, replacements: dict) -> dict:
    """
    Return a shallow-copied attrs dict with placeholders replaced.

    This helper replaces occurrences of placeholders like '{species}' and
    '{direction}' in string values using simple str.replace calls. It does
    not use str.format to avoid errors when templates contain braces for
    other reasons.

    Parameters
    ----------
    attrs : dict
        The attributes dictionary to process (string values may contain
        placeholders).
    replacements : dict
        Mapping of placeholder names (without braces) to replacement values.

    Returns
    -------
    dict
        New attributes dict with replacements applied to string values.
    """
    if not isinstance(attrs, dict):
        return attrs
    new = {}
    for k, v in attrs.items():
        if isinstance(v, str):
            s = v
            for name, val in replacements.items():
                if val is None:
                    continue
                s = s.replace(f"{{{name}}}", str(val))
            new[k] = s
        else:
            new[k] = v
    return new

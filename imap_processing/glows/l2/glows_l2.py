"""Module for GLOWS Level 2 processing."""

import dataclasses
import logging

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.glows import FLAG_LENGTH
from imap_processing.glows.l1b.glows_l1b_data import (
    PipelineSettings,
)
from imap_processing.glows.l2.glows_l2_data import HistogramL2
from imap_processing.glows.utils.constants import GlowsConstants
from imap_processing.spice.time import (
    et_to_datetime64,
    met_to_utc,
    ttj2000ns_to_et,
    ttj2000ns_to_met,
)

logger = logging.getLogger(__name__)


def _pad_daily_lightcurve_bins(value: object, fillval: object) -> np.ndarray:
    """
    Pad chopped daily-lightcurve bin data back to the standard bin count.

    Parameters
    ----------
    value : object
        Chopped daily-lightcurve bin data.
    fillval : object
        CDF fill value used for padded bins.

    Returns
    -------
    numpy.ndarray
        Bin data padded to the standard bin count.
    """
    value_array = np.asarray(value)
    padded_dtype = value_array.dtype
    fillval_dtype = np.asarray(fillval).dtype

    if np.issubdtype(padded_dtype, np.integer):
        try:
            cast_fillval = np.array(fillval, dtype=padded_dtype).item()
        except (OverflowError, TypeError, ValueError):
            padded_dtype = np.result_type(padded_dtype, fillval_dtype)
        else:
            if cast_fillval != fillval:
                padded_dtype = np.result_type(padded_dtype, fillval_dtype)

    padded = np.full(GlowsConstants.STANDARD_BIN_COUNT, fillval, dtype=padded_dtype)
    padded[: len(value_array)] = value_array
    return padded


def glows_l2(
    input_dataset: xr.Dataset,
    pipeline_settings_dataset: xr.Dataset,
    calibration_dataset: xr.Dataset,
) -> list[xr.Dataset]:
    """
    Will process GLOWS L2 data from L1 data.

    Parameters
    ----------
    input_dataset : xarray.Dataset
        Input L1B dataset.
    pipeline_settings_dataset : xarray.Dataset
        Dataset containing pipeline settings from
        GlowsAncillaryCombiner.
    calibration_dataset : xarray.Dataset
        Dataset containing calibration data from
        GlowsAncillaryCombiner.

    Returns
    -------
    list[xarray.Dataset]
        Glows L2 Dataset.
    """
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("glows")
    cdf_attrs.add_instrument_variable_attrs("glows", "l2")

    # Select pipeline settings for the day matching
    # the first epoch in the L1B data. Convert from
    # TT J2000 nanoseconds to datetime64 for indexing.
    day = et_to_datetime64(ttj2000ns_to_et(input_dataset["epoch"].data[0]))
    pipeline_settings = PipelineSettings(
        pipeline_settings_dataset.sel(epoch=day, method="nearest")
    )

    l2 = HistogramL2(input_dataset, pipeline_settings, calibration_dataset)
    if l2.number_of_good_l1b_inputs == 0:
        logger.warning("No good data found in L1B dataset. Returning empty list.")
        return []
    elif (
        np.all(l2.daily_lightcurve.photon_flux == 0)
        and np.all(l2.daily_lightcurve.flux_uncertainties == 0)
        and np.all(l2.daily_lightcurve.exposure_times == 0)
    ):
        logger.warning("All flux and exposure times are zero. Returning empty list.")
        return []
    else:
        return [create_l2_dataset(l2, cdf_attrs, input_dataset.attrs)]


def create_l2_dataset(
    histogram_l2: HistogramL2, attrs: ImapCdfAttributes, input_attrs: dict
) -> xr.Dataset:
    """
    Create a xarray dataset from a HistogramL2 dataclass.

    This dataset should include all the CDF attributes.

    Parameters
    ----------
    histogram_l2 : HistogramL2
        L2 data.
    attrs : ImapCdfAttributes
        CDF attributes for GLOWS L2.
    input_attrs : dict
        Global attributes from the input L1B dataset to propagate.

    Returns
    -------
    xarray.Dataset
        L2 dataset for output to CDF file.
    """
    # Each L2 file only has one timestamp: the midpoint between start and end time.
    time_data = np.array(
        [(histogram_l2.start_time + histogram_l2.end_time) / 2], dtype=np.float64
    )
    # TODO: Create CDF attributes
    epoch_time = xr.DataArray(
        time_data,
        name="epoch",
        dims=["epoch"],
        attrs=attrs.get_variable_attributes("epoch", check_schema=False),
    )

    bins = xr.DataArray(
        np.arange(GlowsConstants.STANDARD_BIN_COUNT, dtype=np.uint32),
        name="bins",
        dims=["bins"],
        attrs=attrs.get_variable_attributes("bins_dim", check_schema=False),
    )

    bins_label = xr.DataArray(
        -1,
        name="bins_label",
        attrs=attrs.get_variable_attributes("bins_label", check_schema=False),
    )

    flags = xr.DataArray(
        np.ones(FLAG_LENGTH, dtype=np.uint8),
        dims=["flags"],
        attrs=attrs.get_variable_attributes("flags_dim", check_schema=False),
    )

    flags_label = xr.DataArray(
        -1,
        name="flags_label",
        attrs=attrs.get_variable_attributes("flags_label", check_schema=False),
    )

    eclipic_data = xr.DataArray(
        np.arange(3, dtype=np.uint8),
        name="ecliptic",
        dims=["ecliptic"],
        attrs=attrs.get_variable_attributes("ecliptic_dim", check_schema=False),
    )

    output = xr.Dataset(
        data_vars={"bins_label": bins_label, "flags_label": flags_label},
        coords={
            "epoch": epoch_time,
            "bins": bins,
            "flags": flags,
            "ecliptic": eclipic_data,
        },
        attrs=attrs.get_global_attributes("imap_glows_l2_hist"),
    )

    output.attrs["flight_software_version"] = input_attrs.get(
        "flight_software_version", ""
    )
    output.attrs["pkts_file_name"] = input_attrs.get("pkts_file_name", [])

    ecliptic_variables = [
        "spacecraft_location_average",
        "spacecraft_location_std_dev",
        "spacecraft_velocity_average",
        "spacecraft_velocity_std_dev",
    ]

    longitudinal_variables = [
        "spin_axis_orientation_average",
        "spin_axis_orientation_std_dev",
    ]

    utc_time_variables = ["start_time", "end_time"]

    for key, value in dataclasses.asdict(histogram_l2).items():
        if key in ecliptic_variables:
            output[key] = xr.DataArray(
                value,
                dims=["epoch", "ecliptic"],
                attrs=attrs.get_variable_attributes(key),
            )
        elif key in longitudinal_variables:
            output[key] = xr.DataArray(
                value,
                dims=["epoch", "latitudinal"],
                attrs=attrs.get_variable_attributes(key),
            )
        elif key == "bad_time_flag_occurrences":
            output[key] = xr.DataArray(
                value,
                dims=["epoch", "flags"],
                attrs=attrs.get_variable_attributes(key),
            )
        elif key in utc_time_variables:
            # Convert time to UTC
            utc_string = [met_to_utc(ttj2000ns_to_met(value))]
            output[key] = xr.DataArray(
                utc_string,
                dims=["epoch"],
                attrs=attrs.get_variable_attributes(key),
            )
        elif key != "daily_lightcurve":
            val = value
            if type(value) is not np.ndarray:
                val = np.array([value])
            output[key] = xr.DataArray(
                val,
                dims=["epoch"],
                attrs=attrs.get_variable_attributes(key),
            )

    for key, value in dataclasses.asdict(histogram_l2.daily_lightcurve).items():
        if key == "number_of_bins":
            # number_of_bins does not have a bins dimension.
            output[key] = xr.DataArray(
                np.array([value], dtype=np.uint16),
                dims=["epoch"],
                attrs=attrs.get_variable_attributes(key),
            )
        else:
            # Bin arrays are chopped to number_of_bins in DailyLightcurve to
            # avoid operating on FILLVAL data. Re-expand to STANDARD_BIN_COUNT
            # here, filling unused bins with the variable's CDF FILLVAL.
            var_attrs = attrs.get_variable_attributes(key)
            padded = _pad_daily_lightcurve_bins(value, var_attrs["FILLVAL"])
            output[key] = xr.DataArray(
                np.array([padded]),
                dims=["epoch", "bins"],
                attrs=var_attrs,
            )

    return output

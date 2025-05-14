"""Methods for processing GLOWS L1B data."""

import dataclasses

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.glows import FLAG_LENGTH
from imap_processing.glows.l1b.glows_l1b_data import DirectEventL1B, HistogramL1B


def glows_l1b(input_dataset: xr.Dataset) -> xr.Dataset:
    """
    Will process the GLOWS L1B data and format the output datasets.

    Parameters
    ----------
    input_dataset : xr.Dataset
        Dataset of input values.

    Returns
    -------
    output_dataset : xr.Dataset
        L1b output dataset.
    """
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("glows")
    cdf_attrs.add_instrument_variable_attrs("glows", "l1b")

    logical_source = (
        input_dataset.attrs["Logical_source"][0]
        if isinstance(input_dataset.attrs["Logical_source"], list)
        else input_dataset.attrs["Logical_source"]
    )

    if "hist" in logical_source:
        output_dataset = create_l1b_hist_output(input_dataset, cdf_attrs)

    elif "de" in logical_source:
        output_dataset = create_l1b_de_output(input_dataset, cdf_attrs)

    else:
        raise ValueError(
            f"Logical_source {input_dataset.attrs['Logical_source']} for input file "
            f"does not match histogram "
            "('hist') or direct event ('de')."
        )

    return output_dataset


def process_de(l1a: xr.Dataset) -> tuple[xr.DataArray]:
    """
    Will process the direct event data from the L1A dataset and return the L1B dataset.

    This method takes in a dataset and applies a function along each timestamp.
    The function here is the creation of a DirectEventL1B object from the data. Each
    variable is passed in as an argument to the lambda function, and then a
    DirectEventL1B object is created from those arguments. This way, each timestamp
    gets its own DirectEventL1B object.

    Parameters
    ----------
    l1a : xr.Dataset
        The L1A dataset to process.

    Returns
    -------
    l1b_arrays : tuple[xr.DataArray]
        The DataArrays for each variable in the L1B dataset.
    """
    dataarrays = [l1a[i] for i in l1a.keys()]

    # Set the dimensions for the input and output dataarrays
    # The dimension include all the non-epoch dimensions. Epoch is the dimension that
    # the data is processed along - so the data from *dataarrays for each timestamp
    # is passed into the function (here a lambda.)

    # We need to specify the other dimensions for input and output so the arrays are
    # properly aligned. The input dimensions are in `input_dims` and the output
    # dimensions are in `output_dims`.

    # An empty array passes the epoch dimension through
    input_dims: list = [[] for i in l1a.keys()]

    output_dimension_mapping = {
        "de_flags": ["flag_dim"],
        "direct_event_glows_times": ["within_the_second"],
        "direct_event_pulse_lengths": ["within_the_second"],
    }

    # For each attribute, retrieve the dims from output_dimension_mapping or use an
    # empty list. Output_dims should be the same length as the number of attributes in
    # the class.
    output_dims = [
        output_dimension_mapping.get(field.name, [])
        for field in dataclasses.fields(DirectEventL1B)
    ]

    # Set the two direct event dimensions. This is the only multi-dimensional L1A
    # (input) variable.
    input_dims[0] = ["within_the_second", "direct_event_components"]

    l1b_fields: tuple = xr.apply_ufunc(
        lambda *args: tuple(dataclasses.asdict(DirectEventL1B(*args)).values()),
        *dataarrays,
        input_core_dims=input_dims,
        output_core_dims=output_dims,
        vectorize=True,
        keep_attrs=True,
    )

    return l1b_fields


def process_histogram(l1a: xr.Dataset) -> xr.Dataset:
    """
    Will process the histogram data from the L1A dataset and return the L1B dataset.

    This method takes in a dataset and applies a function along each timestamp.
    The function here is the creation of a HistogramL1B object from the data. Each
    variable is passed in as an argument to the lambda function, and then a
    HistogramL1B object is created from those arguments. This way, each timestamp
    gets its own HistogramL1B object.

    Parameters
    ----------
    l1a : xr.Dataset
        The L1A dataset to process.

    Returns
    -------
    l1b_arrays : tuple[xr.DataArray]
        The DataArrays for each variable in the L1B dataset. These can be assembled
        directly into a DataSet with the appropriate attributes.
    """
    dataarrays = [l1a[i] for i in l1a.keys()]

    input_dims: list = [[] for i in l1a.keys()]

    # This should include a mapping to every dimension in the output data besides epoch.
    # Only non-1D variables need to be in this mapping.
    output_dimension_mapping = {
        "histogram": ["bins"],
        "imap_spin_angle_bin_cntr": ["bins"],
        "histogram_flag_array": ["bad_angle_flags", "bins"],
        "spacecraft_location_average": ["ecliptic"],
        "spacecraft_location_std_dev": ["ecliptic"],
        "spacecraft_velocity_average": ["ecliptic"],
        "spacecraft_velocity_std_dev": ["ecliptic"],
        "flags": ["flag_dim"],
    }

    # For each attribute, retrieve the dims from output_dimension_mapping or use an
    # empty list. Output_dims should be the same length as the number of attributes in
    # the class.
    output_dims = [
        output_dimension_mapping.get(field.name, [])
        for field in dataclasses.fields(HistogramL1B)
    ]

    # histograms is the only multi dimensional input variable, so we set the non-epoch
    # dimension ("bins").
    # The rest of the input vars are epoch only, so they have an empty list.
    input_dims[0] = ["bins"]

    l1b_fields = xr.apply_ufunc(
        lambda *args: HistogramL1B(*args).output_data(),
        *dataarrays,
        input_core_dims=input_dims,
        output_core_dims=output_dims,
        vectorize=True,
        keep_attrs=True,
    )

    # This is a tuple of dataarrays and not a dataset yet
    return l1b_fields


def create_l1b_hist_output(
    input_dataset: xr.Dataset, cdf_attrs: ImapCdfAttributes
) -> xr.Dataset:
    """
    Create the output dataset for the L1B histogram data.

    This function processes the input dataset and creates a new dataset with the
    appropriate attributes and data variables. It uses the `process_histogram` function
    to process the histogram data.

    Parameters
    ----------
    input_dataset : xr.Dataset
        The input L1A GLOWS Histogram dataset to process.
    cdf_attrs : ImapCdfAttributes
        The CDF attributes to use for the output dataset.

    Returns
    -------
    output_dataset : xr.Dataset
        The output dataset with the processed histogram data and all attributes.
    """
    data_epoch = input_dataset["epoch"]
    data_epoch.attrs = cdf_attrs.get_variable_attributes("epoch", check_schema=False)

    flag_data = xr.DataArray(
        np.arange(FLAG_LENGTH),
        name="bad_time_flags",
        dims=["bad_time_flags"],
        attrs=cdf_attrs.get_variable_attributes(
            "bad_time_flag_hist_attrs", check_schema=False
        ),
    )

    bad_flag_data = xr.DataArray(
        np.arange(4),
        name="bad_angle_flags",
        dims=["bad_angle_flags"],
        attrs=cdf_attrs.get_variable_attributes(
            "bad_angle_flags_attrs", check_schema=False
        ),
    )

    # TODO: the four spacecraft location/velocity values should probably each get
    # their own dimension/attributes
    eclipic_data = xr.DataArray(
        np.arange(3),
        name="ecliptic",
        dims=["ecliptic"],
        attrs=cdf_attrs.get_variable_attributes("ecliptic_attrs", check_schema=False),
    )

    bin_data = xr.DataArray(
        input_dataset["bins"].data,
        name="bins",
        dims=["bins"],
        attrs=cdf_attrs.get_variable_attributes("bins_attrs", check_schema=False),
    )

    bin_label = xr.DataArray(
        bin_data.data.astype(str),
        name="bins_label",
        dims=["bins_label"],
        attrs=cdf_attrs.get_variable_attributes("bins_label", check_schema=False),
    )

    output_dataarrays = process_histogram(input_dataset)

    output_dataset = xr.Dataset(
        coords={
            "epoch": data_epoch,
            "bins": bin_data,
            "bins_label": bin_label,
            "bad_angle_flags": bad_flag_data,
            "bad_time_flags": flag_data,
            "ecliptic": eclipic_data,
        },
        attrs=cdf_attrs.get_global_attributes("imap_glows_l1b_hist"),
    )

    # Since we know the output_dataarrays are in the same order as the fields in the
    # HistogramL1B dataclass, we can use dataclasses.fields to get the field names.

    fields = dataclasses.fields(HistogramL1B)
    for index, dataarray in enumerate(output_dataarrays):
        # Dataarray is already an xr.DataArray type, so we can just assign it
        output_dataset[fields[index].name] = dataarray
        output_dataset[fields[index].name].attrs = cdf_attrs.get_variable_attributes(
            fields[index].name
        )

    output_dataset["bins"] = bin_data
    return output_dataset


def create_l1b_de_output(
    input_dataset: xr.Dataset, cdf_attrs: ImapCdfAttributes
) -> xr.Dataset:
    """
    Create the output dataset for the L1B direct event data.

    Parameters
    ----------
    input_dataset : xr.Dataset
        The input dataset to process.
    cdf_attrs : ImapCdfAttributes
        The CDF attributes to use for the output dataset.

    Returns
    -------
    output_dataset : xr.Dataset
        The output dataset with the processed data.
    """
    data_epoch = input_dataset["epoch"]
    data_epoch.attrs = cdf_attrs.get_variable_attributes("epoch", check_schema=False)

    output_dataarrays = process_de(input_dataset)
    within_the_second_data = xr.DataArray(
        input_dataset["within_the_second"],
        name="within_the_second",
        dims=["within_the_second"],
        attrs=cdf_attrs.get_variable_attributes(
            "within_the_second_attrs", check_schema=False
        ),
    )
    # Add the within_the_second label to the xr.Dataset coordinates
    within_the_second_label = xr.DataArray(
        input_dataset["within_the_second"].data.astype(str),
        name="within_the_second_label",
        dims=["within_the_second_label"],
        attrs=cdf_attrs.get_variable_attributes(
            "within_the_second_label", check_schema=False
        ),
    )

    flag_data = xr.DataArray(
        np.arange(11),
        name="flags",
        dims=["flags"],
        attrs=cdf_attrs.get_variable_attributes("flag_de_attrs", check_schema=False),
    )

    output_dataset = xr.Dataset(
        coords={
            "epoch": data_epoch,
            "within_the_second": within_the_second_data,
            "within_the_second_label": within_the_second_label,
            "flags": flag_data,
        },
        attrs=cdf_attrs.get_global_attributes("imap_glows_l1b_de"),
    )
    fields = dataclasses.fields(DirectEventL1B)

    for index, dataarray in enumerate(output_dataarrays):
        # Dataarray is already an xr.DataArray type, so we can just assign it
        output_dataset[fields[index].name] = dataarray
        output_dataset[fields[index].name].attrs = cdf_attrs.get_variable_attributes(
            fields[index].name
        )

    output_dataset["within_the_second"] = within_the_second_data
    output_dataset.attrs["missing_packets_sequence"] = input_dataset.attrs.get(
        "missing_packets_sequence", ""
    )

    return output_dataset

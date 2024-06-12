"""Methods for processing GLOWS L1B data."""

import dataclasses

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.glows.l1b.glows_l1b_data import DirectEventL1B, HistogramL1B


def glows_l1b(input_dataset: xr.Dataset, data_version: str) -> xr.Dataset:
    """Process the GLOWS L1B data and format the output datasets."""
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("glows")
    cdf_attrs.add_instrument_variable_attrs("glows", "l1b")
    cdf_attrs.add_global_attribute("Data_version", data_version)

    data_epoch = xr.DataArray(
        input_dataset["epoch"],
        name="epoch",
        dims=["epoch"],
        attrs=cdf_attrs.get_variable_attributes("epoch_attrs"),
    )

    logical_source = (
        input_dataset.attrs["Logical_source"][0]
        if isinstance(input_dataset.attrs["Logical_source"], list)
        else input_dataset.attrs["Logical_source"]
    )

    if "hist" in logical_source:
        flag_data = xr.DataArray(
            np.arange(17),
            name="flags",
            dims=["flags"],
            attrs=cdf_attrs.get_variable_attributes("flag_attrs"),
        )
        bad_flag_data = xr.DataArray(
            np.arange(4),
            name="bad_angle_flags",
            dims=["bad_angle_flags"],
            attrs=cdf_attrs.get_variable_attributes("flag_attrs"),
        )

        # TODO: the four spacecraft location/velocity values should probably each get
        # their own dimension/attributes
        eclipic_data = xr.DataArray(
            np.arange(3),
            name="ecliptic",
            dims=["ecliptic"],
            attrs=cdf_attrs.get_variable_attributes("ecliptic_attrs"),
        )

        bin_data = xr.DataArray(
            input_dataset["bins"],
            name="bins",
            dims=["bins"],
            attrs=cdf_attrs.get_variable_attributes("bin_attrs"),
        )

        output_dataarrays = process_histogram(input_dataset)
        # TODO: Is it ok to copy the dimensions from the input dataset?

        output_dataset = xr.Dataset(
            coords={
                "epoch": data_epoch,
                "bins": bin_data,
                "bad_angle_flags": bad_flag_data,
                "flags": flag_data,
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
            output_dataset[
                fields[index].name
            ].attrs = cdf_attrs.get_variable_attributes(fields[index].name)

    elif "de" in logical_source:
        output_dataarrays = process_de(input_dataset)
        per_second_data = xr.DataArray(
            input_dataset["per_second"],
            name="per_second",
            dims=["per_second"],
            attrs=cdf_attrs.get_variable_attributes("per_second_attrs"),
        )

        flag_data = xr.DataArray(
            np.arange(11),
            name="flags",
            dims=["flags"],
            attrs=cdf_attrs.get_variable_attributes("flag_attrs"),
        )

        output_dataset = xr.Dataset(
            coords={
                "epoch": data_epoch,
                "per_second": per_second_data,
                "flags": flag_data,
            },
            attrs=cdf_attrs.get_global_attributes("imap_glows_l1b_de"),
        )
        fields = dataclasses.fields(DirectEventL1B)

        for index, dataarray in enumerate(output_dataarrays):
            # Dataarray is already an xr.DataArray type, so we can just assign it
            output_dataset[fields[index].name] = dataarray
            output_dataset[
                fields[index].name
            ].attrs = cdf_attrs.get_variable_attributes(fields[index].name)

    else:
        raise ValueError(
            f"Logical_source {input_dataset.attrs['Logical_source']} for input file "
            f"does not match histogram "
            "('hist') or direct event ('de')."
        )

    return output_dataset


def process_de(l1a: xr.Dataset) -> tuple[xr.DataArray]:
    """
    Process the direct event data from the L1A dataset and return the L1B dataset.

    This method takes in a dataset and applies a function along each timestamp.
    The function here is the creation of a DirectEventL1B object from the data. Each
    variable is passed in as an argument to the lambda function, and then a
    DirectEventL1B object is created from those arguments. This way, each timestamp
    gets its own DirectEventL1B object.

    Parameters
    ----------
    l1a : xr.Dataset
        The L1A dataset to process

    Returns
    -------
    l1b_arrays: tuple[xr.DataArray]
        The DataArrays for each variable in the L1B dataset
    """
    dataarrays = [l1a[i] for i in l1a.keys()]

    # Set the dimensions for the input and output dataarrays
    # The dimension include all the non-epoch dimensions. Epoch is the dimension that
    # the data is processed along - so the data from *dataarrays for each timestamp
    # is passed into the function (here a lambda.)

    # We need to specify the other dimensions for input and output so the arrays are
    # properly aligned. The input dimensions are in `dims` and the output dimensions are
    # in `new_dims`.

    # An empty array passes the epoch dimension through
    dims = [[] for i in l1a.keys()]
    new_dims = [[] for i in range(13)]

    # Set the two direct event dimensions. This is the only multi-dimensional L1A
    # (input) variable.
    dims[0] = ["per_second", "direct_event"]

    # Flags is a constant length. It has a dimension of "flags"
    new_dims[-3] = ["flags"]

    # glows_times and pulse_lengths should be dimension of "per_second", the same as
    # the input.
    new_dims[-2] = ["per_second"]
    new_dims[-1] = ["per_second"]

    l1b_fields = xr.apply_ufunc(
        lambda *args: tuple(dataclasses.asdict(DirectEventL1B(*args)).values()),
        *dataarrays,
        input_core_dims=dims,
        output_core_dims=new_dims,
        vectorize=True,
        keep_attrs=True,
    )

    return l1b_fields


def process_histogram(l1a: xr.Dataset) -> xr.Dataset:
    """
    Process the histogram data from the L1A dataset and return the L1B dataset.

    This method takes in a dataset and applies a function along each timestamp.
    The function here is the creation of a HistogramL1B object from the data. Each
    variable is passed in as an argument to the lambda function, and then a
    HistogramL1B object is created from those arguments. This way, each timestamp
    gets its own HistogramL1B object.

    Parameters
    ----------
    l1a : xr.Dataset
        The L1A dataset to process

    Returns
    -------
    l1b_arrays: tuple[xr.DataArray]
        The DataArrays for each variable in the L1B dataset. These can be assembled
        directly into a DataSet with the appropriate attributes.
    """
    dataarrays = [l1a[i] for i in l1a.keys()]

    dims = [[] for i in l1a.keys()]
    # 35 is the number of output attributes in the HistogramL1B class.
    new_dims = [[] for i in range(34)]

    # histograms is the only multi dimensional variable, so we need to set its dims to
    # pass along all the dims EXCEPT for epoch. (in this case just "bins")
    # The rest of the vars are epoch only, so they have an empty list.
    dims[0] = ["bins"]

    # This preserves the dimensions for the histogram output
    new_dims[0] = ["bins"]
    new_dims[21] = ["bins"]  # For imap_spin_angle_center - a per-bin value

    # For histogram_flag_array - varies by the new dimension "flags" and by the number
    # of bins
    new_dims[22] = ["bad_angle_flags", "bins"]

    # For the new arrays added: add their dimensions. These aren't defined anywhere,
    # so when the dataset is created I will need to add "ecliptic" as a new dimension.

    # These represent the spacecraft position and std dev, and velocity and std dev.
    new_dims[-5] = ["ecliptic"]
    new_dims[-4] = ["ecliptic"]
    new_dims[-3] = ["ecliptic"]
    new_dims[-2] = ["ecliptic"]

    # Flags is a constant length (17). It has a dimension of "flags"
    new_dims[-1] = ["flags", "bins"]

    l1b_fields = xr.apply_ufunc(
        lambda *args: HistogramL1B(*args).output_data(),
        *dataarrays,
        input_core_dims=dims,
        output_core_dims=new_dims,
        vectorize=True,
        keep_attrs=True,
    )

    # This is a tuple of dataarrays and not a dataset yet
    return l1b_fields

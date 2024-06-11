"""Methods for processing GLOWS L1B data."""

import dataclasses

import xarray as xr

from imap_processing.glows.l1b.glows_l1b_data import DirectEventL1B, HistogramL1B


def glows_l1b_histograms(input_dataset: xr.Dataset) -> xr.Dataset:
    """Process the GLOWS L1B data and format the histogram attribute outputs."""
    # TODO: add CDF attribute steps

    # Will call process_histograms and construct the dataset with the return
    raise NotImplementedError


def glows_l1b_de(input_dataset: xr.Dataset) -> xr.Dataset:
    """Process the GLOWS L1B data and format the direct event attribute outputs."""
    # TODO: generate dataset with CDF attributes
    # Will call process_de and construct the dataset with the return
    raise NotImplementedError


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
    new_dims = [[] for i in range(14)]

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
    # 34 is the number of output attributes in the HistogramL1B class.
    new_dims = [[] for i in range(34)]

    # histograms is the only multi dimensional variable, so we need to set its dims to
    # pass along all the dims EXCEPT for epoch. (in this case just "bins")
    # The rest of the vars are epoch only, so they have an empty list.
    dims[0] = ["bins"]

    # This preserves the dimensions for the histogram output
    new_dims[0] = ["bins"]
    new_dims[22] = ["bins"]  # For imap_spin_angle_center - a per-bin value

    # For histogram_flag_array - varies by the new dimension "flags" and by the number
    # of bins
    new_dims[23] = ["flags", "bins"]

    # For the new arrays added: add their dimensions. These aren't defined anywhere,
    # so when the dataset is created I will need to add "ecliptic" as a new dimension.

    # These represent the spacecraft position and std dev, and velocity and std dev.
    new_dims[-4] = ["ecliptic"]
    new_dims[-3] = ["ecliptic"]
    new_dims[-2] = ["ecliptic"]
    new_dims[-1] = ["ecliptic"]

    # TODO: validate the input dims line up with the dims value
    l1b_fields = xr.apply_ufunc(
        lambda *args: tuple(dataclasses.asdict(HistogramL1B(*args)).values()),
        *dataarrays,
        input_core_dims=dims,
        output_core_dims=new_dims,
        vectorize=True,
    )

    # This is a tuple of dataarrays and not a dataset yet
    return l1b_fields

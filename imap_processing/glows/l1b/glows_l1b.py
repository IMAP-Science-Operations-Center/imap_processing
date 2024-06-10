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


def process_de(l1a: xr.Dataset) -> xr.Dataset:
    """Process the direct event data from the L1A dataset and return the L1B dataset."""
    dataarrays = [l1a[i] for i in l1a.keys()]

    # An empty array passes the epoch dimension through
    dims = [[] for i in l1a.keys()]
    new_dims = [[] for i in range(14)]

    # Set the two DE dimensions. This is the only multi-dimensional L1A variable.
    dims[0] = ["per_second", "direct_event"]

    # Flags is a constant length
    new_dims[-3] = ["flags"]
    # glows_times and pulse_lengths should be length of "per_second"
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
    """Process the histogram data from the L1A dataset and return the L1B dataset."""
    dataarrays = [l1a[i] for i in l1a.keys()]

    dims = [[] for i in l1a.keys()]
    # 32 is the number of attributes in the HistogramL1B class.
    new_dims = [[] for i in range(34)]

    # histograms is the only multi dimensional variable, so we need to set its dims to
    # pass along all the dims EXCEPT for epoch. (in this case just "bins")
    # The rest of the vars are only epoch so they have an empty list.
    dims[0] = ["bins"]

    # This preserves the dimensions
    new_dims[0] = ["bins"]
    new_dims[22] = ["bins"]  # flags need another dimension
    new_dims[23] = ["flags", "bins"]

    # For the new arrays added: add their dimensions. These aren't defined anywhere,
    # so when the dataset is created I will need to add "coords" as a new dimension.
    new_dims[-4] = ["coords"]
    new_dims[-3] = ["coords"]
    new_dims[-2] = ["coords"]
    new_dims[-1] = ["coords"]

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

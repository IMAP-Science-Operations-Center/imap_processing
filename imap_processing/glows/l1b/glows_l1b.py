"""Methods for processing GLOWS L1B data."""

import dataclasses

import xarray as xr

from imap_processing.glows.l1b.glows_l1b_data import HistogramL1B


def read_input_cdf(input_file):
    """
    Read the input CDF file and return the data as a list of GLOWS L1A products.

    Parameters
    ----------
    input_file : str
        The path to the input CDF file.
    """
    pass


def glows_l1b():
    """Process the GLOWS L1B data."""
    # read in the data from CDF files

    # For each dataarray, map each point -> histogram output

    # then, write a method to reduce list of histogram input ->  one dataset output
    pass


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

    l1b_fields = xr.apply_ufunc(
        histogram_mapping,
        # TODO rearrange class to match with dataset
        *dataarrays,
        input_core_dims=dims,
        output_core_dims=new_dims,
        vectorize=True,
    )

    # This is a tuple of dataarrays and not a dataset yet
    return l1b_fields


def histogram_mapping(*args) -> tuple:
    """For each variable in the L1A dataset, pass that into the HistogramL1B class.

    This class automatically converts all L1A values to L1B values and creates
    additional data fields. Then, the return is just the data fields of the L1B class,
    so the output file exactly lines up with that class.

    Attributes
    ----------
    args : tuple
        The args should exactly line up with the init for the HistogramL1B class.

    Returns
    -------
    tuple
        The data fields of the HistogramL1B class.
    """
    # for arg in args:
    # print(f"Arg type: {type(arg)}\n")
    # print(arg)
    # Given: all the inputs for histogramL1B init
    # Return: a tuple of all the histogramL1B attributes

    # TODO: validate expected types and number of inputs, so we can have a more
    #  intelligent error?
    hist_l1b = HistogramL1B(*args)

    return tuple(dataclasses.asdict(hist_l1b).values())
    # might need to change input - a list of dataarrays?

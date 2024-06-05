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
    print(f"Dataarray len: {len(dataarrays)}")
    print(f"Dataarrays type: {type(dataarrays[0])}")

    dims = [[] for i in l1a.keys()]
    new_dims = [[] for i in range(28)]

    # This is the only multi dimensional variable, so we need to set the dims to be all
    # the dims EXCEPT for epoch.
    # The rest of the vars are only epoch so they have an empty list.
    dims[0] = ["bins"]
    # TODO: new dims needs to include all the ndarray types as well - do we add a dim
    #  for 3 value arr
    new_dims[0] = ["bins"]
    l1b_fields = xr.apply_ufunc(
        histogram_mapping,
        # TODO rearrange class to match with dataset
        *dataarrays,
        input_core_dims=dims,
        output_core_dims=new_dims,
        vectorize=True,
    )
    print("done")
    return l1b_fields


def histogram_mapping(*args) -> tuple:
    """Do things. (ruff is being annoying)."""
    for arg in args:
        print(f"Arg type: {type(arg)}\n")
        print(arg)
    # Given: all the inputs for histogramL1B init
    # Return: a tuple of all the histogramL1B attributes

    # TODO: validate expected types and number of inputs, so we can have a more
    #  intelligent error?
    hist_l1b = HistogramL1B(*args)
    print("Generated hist_l1b")

    return tuple(dataclasses.asdict(hist_l1b).values())
    # might need to change input - a list of dataarrays?

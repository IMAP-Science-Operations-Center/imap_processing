import dataclasses

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.glows.l1b.glows_l1b_data import HistogramL1B
from imap_processing.glows.l2.glows_l2_data import HistogramL2


def glows_l2(input_dataset: xr.Dataset, data_version: str) -> xr.Dataset:
    """
    Will process GLoWS L2 data from L1 data.

    Parameters
    ----------
    input_dataset
    data_version

    Returns
    -------

    """
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("glows")
    cdf_attrs.add_instrument_variable_attrs("glows", "l2")
    cdf_attrs.add_global_attribute("Data_version", data_version)

    split_data = split_data_by_observational_day(input_dataset)
    l2_values = []
    for data in split_data:
        l2 = generate_l2(data)
        l2_values.append(create_l2_dataset(l2, cdf_attrs))
    # todo: what does epoch mean here? Just one epoch value?

    # TODO: eliminate bad histograms per-epoch time based on l1b.flags using a mask with
    #  "active bad times" in ancillary file

    # TODO: compute averages for the range of L1B instances


# TODO: filter good times out
def generate_l2(l1b_dataset: xr.Dataset) -> HistogramL2:
    # most of the values from L1B are averaged over a day

    # one dataset collects multiple epoch values which need to be averaged down into
    # one value.
    all_variables = dataclasses.fields(HistogramL1B)

    # Generate outputs that are passed in directly from L1B
    var_outputs = {'total_l1b_inputs': len(l1b_dataset['epoch']),
                   'number_of_good_l1b_inputs': len(l1b_dataset['epoch']),
                   # TODO replace post-filter
                   # 'identifier': 'test', # TODO: retrieve from unique_block_identifier
                   # TODO: start and end time should be only in good times
                   'start_time': l1b_dataset['epoch'].data[0],
                   'end_time': l1b_dataset['epoch'].data[-1],
                   'histogram': l1b_dataset['histogram'].data,
                   # TODO is this type correct?
                   'bad_time_flag_occurrences': None,
                   'flight_software_version':
                       l1b_dataset['flight_software_version'].data[0],
                   }

    for field in all_variables:
        var_name = field.name
        if 'average' in var_name:
            var_outputs[var_name] = l1b_dataset[var_name].mean(dim="epoch").data

        if 'variance' in var_name:
            expected_var = var_name.replace('variance', 'average')
            var_outputs[var_name] = l1b_dataset[expected_var].std(dim="epoch").data

    output = HistogramL2(**var_outputs)

    print(l1b_dataset.data_vars)

    return output


def split_data_by_observational_day(input_dataset: xr.Dataset) -> list[xr.Dataset]:
    """
    Return L1B data array for an observational day, given start and stop times.

    Parameters
    ----------
    input_dataset
        Input L1B dataset

    Returns
    -------
    list[xr.Dataset]
    """
    # Find the range of epoch values within the observational day.
    # This should be replaced with a query to the spin table to get the observational
    # days within the time range of the file and when those observational days
    # start and stop

    day_ends = [
        input_dataset["epoch"].data[0],
        input_dataset["epoch"].data[100],
        input_dataset["epoch"].data[-1],
    ]
    print(day_ends)

    # TODO: this slice is inclusive on the start and end for some reason. When you
    #  replace this slice with the real spin data, figure that out.
    data_by_day = [
        input_dataset.sel(epoch=slice(day_ends[i], day_ends[i + 1]))
        for i in range(len(day_ends) - 1)
    ]
    # print(data_by_day)

    print(data_by_day[0]["epoch"].data[0])
    print(data_by_day[0]["epoch"].data[-1])

    print(data_by_day[1]["epoch"].data[0])
    print(data_by_day[1]["epoch"].data[-1])

    # TODO: collect Histograml1B objects per observational day
    # For now, I guess just use a file to retrieve that info?

    # todo: run avg/std_dev methods on all the avg/std_dev variables
    # then put them and the histograms into the glows_l2 dataclass and run some method
    # to compute lightcurves.
    # then take that and put it into a dataset and return it out.
    return data_by_day


def create_l2_dataset(histogram_l2: HistogramL2,
                      attrs: ImapCdfAttributes) -> xr.Dataset:
    pass

"""MAG L1C processing module."""

import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes


def mag_l1c(
    first_input_dataset: xr.Dataset, second_input_dataset: xr.Dataset, version: str
) -> xr.Dataset:
    """
    Will process MAG L1C data from L1A data.

    This requires both the norm and burst data to be passed in.

    Parameters
    ----------
    first_input_dataset : xr.Dataset
        The first input dataset to process. This can be either burst or norm data, for
        mago or magi.
    second_input_dataset : xr.Dataset
        The second input dataset to process. This should be burst if first_input_dataset
        was norm, or norm if first_input_dataset was burst. It should match the
        instrument - both inputs should be mago or magi.
    version : str
        The version of the output data.

    Returns
    -------
    output_dataset : xr.Dataset
        L1C data set.
    """
    # TODO: L1C processing involves filling gaps with burst data.
    input_logical_source_1 = first_input_dataset.attrs["Logical_source"]
    if isinstance(first_input_dataset.attrs["Logical_source"], list):
        input_logical_source_1 = first_input_dataset.attrs["Logical_source"][0]

    input_logical_source_2 = second_input_dataset.attrs["Logical_source"]
    if isinstance(second_input_dataset.attrs["Logical_source"], list):
        input_logical_source_2 = second_input_dataset.attrs["Logical_source"][0]

    if "norm" in input_logical_source_1:
        output_dataset = first_input_dataset.copy()
        logical_source = input_logical_source_1.replace("l1b", "l1c")
    elif "norm" in input_logical_source_2:
        output_dataset = second_input_dataset.copy()
        logical_source = input_logical_source_2.replace("l1b", "l1c")
    else:
        raise RuntimeError("Neither input dataset is norm data")

    attribute_manager = ImapCdfAttributes()
    attribute_manager.add_instrument_global_attrs("mag")
    attribute_manager.add_global_attribute("Data_version", version)

    output_dataset.attrs = attribute_manager.get_global_attributes(logical_source)

    return output_dataset

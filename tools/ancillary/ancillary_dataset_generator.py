""" File for taking in multiple ancillary files and creating a dataset over the required time range."""
from collections import namedtuple
from datetime import datetime

from abc import ABC, abstractmethod
from cdflib.xarray import cdf_to_xarray
import xarray as xr

from dataclasses import dataclass
from imap_data_access import AncillaryFilePath
from imap_data_access.processing_input import ProcessingInputCollection, \
    ProcessingInputType, ProcessingInput

DataWithRange = namedtuple("DataWithRange", ["start_time", "end_time", "dataset", "version"])


class AncillaryConverter(ABC):
    time_variable = "epoch"

    def __init__(self, ancillary_input: ProcessingInput):
        """
        Add processing inputs to the dataset
        """
        # TODO: each processing input represents multiple files to combine. Combine them here.
        # Get date range of each file, then set the start times for datasets and versions.
        # Go through each datapoint and recursively:
        # 1. see if any range is already covered in the dataset
        # 2. determine which version number is higher
        # 3. Overwrite the covered ranges with the new data
        # 4. Add any new data to the dataset. Save data into datavars and index of the range, dataset, and version tuple.

        self.dataset_list = []
        # TODO NEXT STEP: WRITE SOME TESTS

        for file in ancillary_input.filename_list:
            self.dataset_list.append(self.convert_to_dataset(file))

        self.combined_dataset = self.combine_datasets(ancillary_input)

    def convert_to_dataset(self, filepath: str) -> xr.Dataset:
        """
        Method for converting the input filepaths to an xarray dataset.
        """
        return cdf_to_xarray(filepath)


    def combine_datasets(self, ancillary_input) -> xr.Dataset:
        """
        Combine all the input datasets into one output dataset.

        This instance works if there is no time-varying data inside the input datasets -
        that is, the output should be a single dataset with
        Returns
        -------

        """
        output_dataset = xr.Dataset()
        data_list = []
        for filename in ancillary_input.filename_list:
            filepath = AncillaryFilePath(filename)
            dataset = self.convert_to_dataset(filename)
            data_with_range = DataWithRange(filepath.start_date, filepath.end_date, dataset, filepath.version)
            data_list = self.add_dataset_to_output(data_list, data_with_range)

        print(f"Final data list: {data_list}")
        #TODO collapse data_list into a single dataset with xr.concat along "epoch" dim
        return self.convert_data_list_to_dataset(data_list)

    def convert_data_list_to_dataset(self, data_list) -> xr.Dataset:
        """

        Parameters
        ----------
        data_list

        Returns
        -------

        """
        datasets = [row[0] for row in data_list]
        output_ds = xr.concat(datasets, dim=self.time_variable)

        return output_ds


    def add_dataset_to_output(self, input_list, data_with_range: DataWithRange):
        """
        Given an input dataset, add the new DataWithRange piece of data and return.

        Overwrite any overlapping data that already exists in input_dataset if the
        version number is higher.

        Parameters
        ----------
        input_dataset: xr.Dataset
            The dataset to add the new data to.
        data_with_range : DataWithRange
            The new data to add to the dataset.

        Returns
        -------

        """

        epoch_timestamp = datetime.strptime(data_with_range.start_time, "%Y%m%d")
        # Check for epoch in existing input_list
        new_epoch = epoch_timestamp
        expanded_datasets = data_with_range.dataset.expand_dims(dim={self.time_variable: [new_epoch]})
        print(f"Adding dataset to input_list with epoch {new_epoch} and version {data_with_range.version}")
        input_list.append([expanded_datasets, new_epoch, data_with_range.version])

        return input_list

class MagAncillaryConverter(AncillaryConverter):
    def __init__(self, ancillary_input: ProcessingInput):
        super().__init__(ancillary_input)

    def convert_to_dataset(self, filepath) -> xr.Dataset:
        return cdf_to_xarray(filepath)




class AncillaryDatasetGenerator:
    """

    """
    # I need to:
    # 1. Read in all files as xarray datasets
    # 2. Using time range and version numbers, overwrite and extend input datasets
    # 3. return the final dataset.
    def __init__(self, input_collection: ProcessingInputCollection, ancillary_converter: AncillaryConverter):
        self.ancillary_inputs = input_collection.get_processing_inputs(input_type=ProcessingInputType.ANCILLARY_FILE)
        # TODO: each processinginput should return one xarray. Only the files in one processinginput should be combined together.
        for ancillary_input in self.ancillary_inputs:
            ancillary_converter(anc)


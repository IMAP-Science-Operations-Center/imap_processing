"""Methods for processing GLOWS L1B data."""

import dataclasses

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.glows import FLAG_LENGTH
from imap_processing.glows.l1b.glows_l1b_data import (
    AncillaryExclusions,
    AncillaryParameters,
    DirectEventL1B,
    HistogramL1B,
    PipelineSettings,
)
from imap_processing.spice.time import et_to_datetime64, ttj2000ns_to_et


def glows_l1b(
    input_dataset: xr.Dataset,
    excluded_regions: xr.Dataset,
    uv_sources: xr.Dataset,
    suspected_transients: xr.Dataset,
    exclusions_by_instr_team: xr.Dataset,
    pipeline_settings_dataset: xr.Dataset,
    conversion_table_dict: dict,
) -> xr.Dataset:
    """
    Will process the histogram GLOWS L1B data and format the output datasets.

    Parameters
    ----------
    input_dataset : xr.Dataset
        Dataset of input values for L1A histogram data.
    excluded_regions : xr.Dataset
        Dataset containing excluded sky regions with ecliptic coordinates. This
        is the output from GlowsAncillaryCombiner.
    uv_sources : xr.Dataset
        Dataset containing UV sources (stars) with coordinates and masking radii. It is
        the output from GlowsAncillaryCombiner.
    suspected_transients : xr.Dataset
        Dataset containing suspected transient signals with time-based masks. This is
        the output from GlowsAncillaryCombiner.
    exclusions_by_instr_team : xr.Dataset
        Dataset containing manual exclusions by instrument team with time-based masks.
        This is the output from GlowsAncillaryCombiner.
    pipeline_settings_dataset : xr.Dataset
        Dataset containing pipeline settings and other ancillary parameters.
    conversion_table_dict : dict
        Dict containing the L1B conversion table for decoding ancillary parameters.
        This is read directly out of the JSON file.

    Returns
    -------
    output_dataset : xr.Dataset
        L1b output dataset.
    """
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("glows")
    cdf_attrs.add_instrument_variable_attrs("glows", "l1b")

    day = et_to_datetime64(ttj2000ns_to_et(input_dataset["epoch"].data[0]))

    # Create ancillary exclusions object from passed-in datasets
    ancillary_exclusions = AncillaryExclusions(
        excluded_regions=excluded_regions,
        uv_sources=uv_sources,
        suspected_transients=suspected_transients,
        exclusions_by_instr_team=exclusions_by_instr_team,
    )
    pipeline_settings = PipelineSettings(
        pipeline_settings_dataset.sel(epoch=day, method="nearest"),
    )

    ancillary_parameters = AncillaryParameters(conversion_table_dict)

    output_dataarrays = process_histogram(
        input_dataset, ancillary_exclusions, ancillary_parameters, pipeline_settings
    )
    output_dataset = create_l1b_hist_output(
        output_dataarrays, input_dataset["epoch"], input_dataset["bins"], cdf_attrs
    )

    return output_dataset


def glows_l1b_de(
    input_dataset: xr.Dataset,
    conversion_table_dict: dict,
) -> xr.Dataset:
    """
    Process GLOWS L1B direct events data.

    Parameters
    ----------
    input_dataset : xr.Dataset
        The input dataset to process.
    conversion_table_dict : dict
        Dict containing the L1B conversion table for decoding ancillary parameters.

    Returns
    -------
    xr.Dataset
        The processed L1B direct events dataset.
    """
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("glows")
    cdf_attrs.add_instrument_variable_attrs("glows", "l1b")

    ancillary_parameters = AncillaryParameters(conversion_table_dict)

    output_dataset = create_l1b_de_output(
        input_dataset, cdf_attrs, ancillary_parameters
    )

    return output_dataset


def process_de(
    l1a: xr.Dataset, ancillary_parameters: AncillaryParameters
) -> tuple[xr.DataArray]:
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
    ancillary_parameters : AncillaryParameters
        The ancillary parameters for decoding DE data.

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

    # Create a closure that captures the ancillary parameters
    def create_direct_event_l1b(*args) -> tuple:  # type: ignore[no-untyped-def]
        """
        Create DirectEventL1B object with captured ancillary parameters.

        Parameters
        ----------
        *args
            Variable arguments passed from xr.apply_ufunc containing L1A data.

        Returns
        -------
        tuple
            Tuple of values from DirectEventL1B dataclass.
        """
        return tuple(
            dataclasses.asdict(
                DirectEventL1B(*args, ancillary_parameters)  # type: ignore[call-arg]
            ).values()
        )

    l1b_fields: tuple = xr.apply_ufunc(
        create_direct_event_l1b,
        *dataarrays,
        input_core_dims=input_dims,
        output_core_dims=output_dims,
        vectorize=True,
        keep_attrs=True,
    )

    return l1b_fields


def process_histogram(
    l1a: xr.Dataset,
    ancillary_exclusions: AncillaryExclusions,
    ancillary_parameters: AncillaryParameters,
    pipeline_settings: PipelineSettings,
) -> xr.Dataset:
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
    ancillary_exclusions : AncillaryExclusions
        The ancillary exclusions data for bad-angle flag processing.
    ancillary_parameters : AncillaryParameters
        The ancillary parameters for decoding histogram data.
    pipeline_settings : PipelineSettings
        The pipeline settings including flag activation.

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
        "spin_axis_orientation_average": ["latitudinal"],
        "spin_axis_orientation_std_dev": ["latitudinal"],
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

    # Create a closure that captures the ancillary objects
    def create_histogram_l1b(*args) -> tuple:  # type: ignore[no-untyped-def]
        """
        Create HistogramL1B object with captured ancillary data.

        Parameters
        ----------
        *args
            Variable arguments passed from xr.apply_ufunc containing L1A data.

        Returns
        -------
        tuple
            Tuple of processed L1B data arrays from HistogramL1B.output_data().
        """
        return HistogramL1B(  # type: ignore[call-arg]
            *args, ancillary_exclusions, ancillary_parameters, pipeline_settings
        ).output_data()

    l1b_fields = xr.apply_ufunc(
        create_histogram_l1b,
        *dataarrays,
        input_core_dims=input_dims,
        output_core_dims=output_dims,
        vectorize=True,
        keep_attrs=True,
    )

    # This is a tuple of dataarrays and not a dataset yet
    return l1b_fields


def create_l1b_hist_output(
    l1b_dataarrays: tuple[xr.DataArray],
    epoch: xr.DataArray,
    bin_coord: xr.DataArray,
    cdf_attrs: ImapCdfAttributes,
) -> xr.Dataset:
    """
    Create the output dataset for the L1B histogram data.

    This function takes in the output from `process_histogram`, which is a tuple of
    DataArrays matching the output L1B data variables, and assembles them into a
    Dataset with the appropriate coordinates.

    Parameters
    ----------
    l1b_dataarrays : tuple[xr.DataArray]
        The DataArrays for each variable in the L1B dataset. These align with the
        fields in the HistogramL1B dataclass, which also describes each variable.
    epoch : xr.DataArray
        The epoch DataArray to use as a coordinate in the output dataset. Generally
        equal to the L1A epoch.
    bin_coord : xr.DataArray
        An arange DataArray for the bins coordinate. Nominally expected to be equal to
        `xr.DataArray(np.arange(number_of_bins_per_histogram), name="bins",
        dims=["bins"])`. Pulled up from L1A.
    cdf_attrs : ImapCdfAttributes
        The CDF attributes to use for the output dataset.

    Returns
    -------
    output_dataset : xr.Dataset
        The output dataset with the processed histogram data and all attributes.
    """
    data_epoch = epoch
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

    latitudinal_data = xr.DataArray(
        np.arange(2),
        name="latitudinal",
        dims=["latitudinal"],
        attrs=cdf_attrs.get_variable_attributes(
            "latitudinal_attrs", check_schema=False
        ),
    )

    bin_data = xr.DataArray(
        bin_coord.data,
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

    output_dataset = xr.Dataset(
        coords={
            "epoch": data_epoch,
            "bins": bin_data,
            "bins_label": bin_label,
            "bad_angle_flags": bad_flag_data,
            "bad_time_flags": flag_data,
            "ecliptic": eclipic_data,
            "latitudinal": latitudinal_data,
        },
        attrs=cdf_attrs.get_global_attributes("imap_glows_l1b_hist"),
    )

    # Since we know the output_dataarrays are in the same order as the fields in the
    # HistogramL1B dataclass, we can use dataclasses.fields to get the field names.

    fields = dataclasses.fields(HistogramL1B)
    for index, dataarray in enumerate(l1b_dataarrays):
        # Dataarray is already an xr.DataArray type, so we can just assign it
        output_dataset[fields[index].name] = dataarray
        output_dataset[fields[index].name].attrs = cdf_attrs.get_variable_attributes(
            fields[index].name
        )

    output_dataset["bins"] = bin_data
    return output_dataset


def create_l1b_de_output(
    input_dataset: xr.Dataset,
    cdf_attrs: ImapCdfAttributes,
    ancillary_parameters: AncillaryParameters,
) -> xr.Dataset:
    """
    Create the output dataset for the L1B direct event data.

    Parameters
    ----------
    input_dataset : xr.Dataset
        The input dataset to process.
    cdf_attrs : ImapCdfAttributes
        The CDF attributes to use for the output dataset.
    ancillary_parameters : AncillaryParameters
        The ancillary parameters for decoding DE data.

    Returns
    -------
    output_dataset : xr.Dataset
        The output dataset with the processed data.
    """
    data_epoch = input_dataset["epoch"]
    data_epoch.attrs = cdf_attrs.get_variable_attributes("epoch", check_schema=False)

    output_dataarrays = process_de(input_dataset, ancillary_parameters)
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

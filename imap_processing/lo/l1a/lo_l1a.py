import xarray as xr
import numpy as np
from dataclasses import fields
from collections import namedtuple
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import write_cdf
from imap_processing.lo.l0.lo_apid import LoAPID
from imap_processing.lo.l0.data_classes.science_direct_events import ScienceDirectEvents
from imap_processing.lo.l0.data_classes.science_counts import ScienceCounts
from imap_processing.lo.l0.data_classes.star_sensor import StarSensor
def lo_l1a(dependency):

    #TODO: decom the CCSDS file

    #TODO: sort the decommed packets by SHCOARSE

    #TODO: group the packets by APID

    #TODO: Loop through packets
    #TODO: Inside Loop: Check for each APID and Initialize data class object with packet contents, add to list

    #TODO: Create dataset
    #TODO: Write CDF

    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="lo", level="l1a")
    # Temporary code. the fields will still be pulled from the data classes,
    # the data class to use will be determined by the APID in the decommed file
    # and the data class will first be populated with the packet data.
    if "_de_" in dependency:
        data_fields = fields(ScienceDirectEvents)
        logical_source = "imap_lo_l1a_de"
    elif "_histogram_" in dependency:
        data_fields = fields(ScienceCounts)
        logical_source = "imap_lo_l1a_histogram"
    elif "_star_" in dependency:
        data_fields = fields(StarSensor)
        logical_source = "imap_lo_l1a_star"
    elif "_spin_" in dependency:
        # Spin data class doesn't exist yet. fake the fields() output
        data_field_tup = namedtuple("data_field_tup", ["name"])
        data_fields = [data_field_tup("NUM_COMPLETED"),
                       data_field_tup("ACQ_END"),
                       data_field_tup("SPIN_SECONDS"),
                       data_field_tup("SPIN_SUBSECONDS"),
                       data_field_tup("ESA_P_DAC"),
                       data_field_tup("ESA_N_DAC"),
                       data_field_tup("VAL_PERIOD"),
                       data_field_tup("VAL_SPIN"),
                       data_field_tup("SOURCE")]
        logical_source = "imap_lo_l1a_spin"

    dataset = create_datasets(attr_mgr, logical_source, data_fields)

    return write_cdf(dataset)


def create_datasets(attr_mgr, logical_source, data_fields):

    # NOTE: At this point, we keep epoch time as raw value from packet
    # which is in seconds and spacecraft time. Some instrument uses this
    # raw value in processing. If you want to convert this to datetime
    # object, you can use `update_epoch_to_datetime` function afterwards.
    epoch_time = xr.DataArray(
        [0, 1, 2],
        name="epoch",
        dims=["epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )

    dataset = xr.Dataset(
        coords={"epoch": epoch_time},
        attrs=attr_mgr.get_global_attributes(logical_source)
    )

    for field in data_fields:
        field = field.name.lower()
        if field not in ["shcoarse", "ground_sw_version", "packet_file_name", "ccsds_header", "data"]:
            print(f"FIELD: {field}")
            dims = [value for key, value in attr_mgr.get_variable_attributes(field).items() if "DEPEND" in key]
            print(f"DIMS: {dims}")
            dataset[field] = xr.DataArray(
                [0, 0, 0],
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field)
            )

    return dataset


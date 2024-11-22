import numpy as np
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.hit.hit_utils import (
    HitAPID,
    concatenate_leak_variables,
    get_attribute_manager,
    get_datasets_by_apid,
    process_housekeeping_data,
)


@pytest.fixture(scope="module")
def packet_filepath():
    """Set path to test data file"""
    return (
        imap_module_directory / "tests/hit/test_data/imap_hit_l0_raw_20100105_v001.pkts"
    )


@pytest.fixture(scope="module")
def attribute_manager():
    """Create the attribute manager"""
    data_version = "001"
    level = "l1a"
    attr_mgr = get_attribute_manager(data_version, level)
    return attr_mgr


@pytest.fixture(scope="module")
def housekeeping_dataset(packet_filepath):
    """Get the housekeeping dataset"""
    # Unpack ccsds file to xarray datasets
    datasets_by_apid = get_datasets_by_apid(packet_filepath)
    return datasets_by_apid[HitAPID.HIT_HSKP]


def test_get_datasets_by_apid(packet_filepath):
    result = get_datasets_by_apid(packet_filepath)

    assert isinstance(result, dict)
    assert HitAPID.HIT_HSKP in result
    # assert HitAPID.HIT_SCIENCE in result


def test_get_attribute_manager():
    data_version = "001"
    level = "l1a"
    attr_mgr = get_attribute_manager(data_version, level)

    assert isinstance(attr_mgr, ImapCdfAttributes)


def test_concatenate_leak_variables(housekeeping_dataset):
    """Test concatenation of leak_i variables"""

    # Create data array for leak_i dependency
    adc_channels = xr.DataArray(
        np.arange(64, dtype=np.uint8),
        name="adc_channels",
        dims=["adc_channels"],
    )

    updated_dataset = concatenate_leak_variables(housekeeping_dataset, adc_channels)

    # Assertions
    # ----------------
    assert "leak_i" in updated_dataset
    assert updated_dataset["leak_i"].shape == (88, 64)
    for i in range(64):
        # Check if the values in the `leak_i` variable match the values in
        # the original `leak_i_XX` variable.
        #  - First access the `leak_i` variable in the `updated_dataset`.
        #    The [:, i] selects all rows (`:`) and the `i`-th column of the `leak_i`
        #    variable.
        #  - Then access the `leak_i_XX` variable in the `housekeeping_dataset`.
        #    The `f"leak_i_{i:02d}"` selects the variable with the name `leak_i_XX`
        #    where `XX` is the `i`-th value.
        #  - Compare values
        np.testing.assert_array_equal(
            updated_dataset["leak_i"][:, i], housekeeping_dataset[f"leak_i_{i:02d}"]
        )


def test_process_housekeeping(housekeeping_dataset, attribute_manager):
    """Test processing of housekeeping dataset"""

    # Call the function
    processed_hskp_dataset = process_housekeeping_data(
        housekeeping_dataset, attribute_manager, "imap_hit_l1a_hk"
    )

    # Define the keys that should have dropped from the dataset
    dropped_keys = {
        "pkt_apid",
        "sc_tick",
        "version",
        "type",
        "sec_hdr_flg",
        "seq_flgs",
        "src_seq_ctr",
        "pkt_len",
        "hskp_spare1",
        "hskp_spare2",
        "hskp_spare3",
        "hskp_spare4",
        "hskp_spare5",
    }
    # Define the keys that should be present
    valid_keys = {
        "heater_on",
        "fsw_version_b",
        "ebox_m12va",
        "phasic_stat",
        "ebox_3d4vd",
        "ebox_p2d0vd",
        "temp1",
        "last_bad_seq_num",
        "ebox_m5d7va",
        "ebox_p12va",
        "table_status",
        "enable_50khz",
        "mram_disabled",
        "temp3",
        "preamp_l1a",
        "l2ab_bias",
        "l34b_bias",
        "fsw_version_c",
        "num_evnt_last_hk",
        "dac1_enable",
        "preamp_l234b",
        "analog_temp",
        "fee_running",
        "fsw_version_a",
        "num_errors",
        "test_pulser_on",
        "dac0_enable",
        "preamp_l1b",
        "l1ab_bias",
        "l34a_bias",
        "leak_i",
        "last_good_cmd",
        "lvps_temp",
        "idpu_temp",
        "temp2",
        "preamp_l234a",
        "last_good_seq_num",
        "num_good_cmds",
        "heater_control",
        "hvps_temp",
        "ebox_p5d7va",
        "spin_period_long",
        "enable_hvps",
        "temp0",
        "spin_period_short",
        "dyn_thresh_lvl",
        "num_bad_cmds",
        "adc_mode",
        "ebox_5d1vd",
        "active_heater",
        "last_error_num",
        "last_bad_cmd",
        "ref_p5v",
        "code_checksum",
        "mode",
    }

    # Define the dataset attributes
    dataset_attrs = {
        "Acknowledgement": "Please acknowledge the IMAP Mission Principal "
        "Investigator, Prof. David J. McComas of Princeton "
        "University.\n",
        "Data_level": "1A",
        "Data_type": "L1A_HK>Level-1A Housekeeping",
        "Data_version": "001",
        "Descriptor": "HIT>IMAP High-energy Ion Telescope",
        "Discipline": "Solar Physics>Heliospheric Physics",
        "File_naming_convention": "source_descriptor_datatype_yyyyMMdd_vNNN",
        "HTTP_LINK": "https://imap.princeton.edu/",
        "Instrument_type": "Particles (space)",
        "LINK_TITLE": "IMAP The Interstellar Mapping and Acceleration Probe",
        "Logical_file_id": None,
        "Logical_source": "imap_hit_l1a_hk",
        "Logical_source_description": "IMAP Mission HIT Instrument Level-1A "
        "Housekeeping Data.",
        "Mission_group": "IMAP",
        "PI_affiliation": "Princeton University",
        "PI_name": "Prof. David J. McComas",
        "Project": "STP>Solar Terrestrial Probes",
        "Rules_of_use": "All IMAP data products are publicly released and citable for "
        "use in publications. Please consult the IMAP team "
        "publications and personnel for further details on "
        "production, processing, and usage of these data.\n",
        "Source_name": "IMAP>Interstellar Mapping and Acceleration Probe",
        "TEXT": "The High-energy Ion Telescope (HIT) measures the elemental "
        "composition, energy spectra, angle distributions, and arrival "
        "times of high-energy ions. HIT delivers full-sky coverage from "
        "a wide instrument field-of-view (FOV) to enable a high resolution "
        "of ion measurements, such as observing shock-accelerated ions, "
        "determining the origin of the solar energetic particles (SEPs) "
        "spectra, and resolving particle transport in the heliosphere. "
        "See https://imap.princeton.edu/instruments/hit for more details.\n",
    }

    # Define the coordinates and dimensions. Both have equivalent values
    dataset_coords_dims = {"epoch", "adc_channels", "adc_channels_label"}

    # Assertions
    # ----------------
    # Check that the dataset has the correct variables
    assert valid_keys == set(processed_hskp_dataset.data_vars.keys())
    assert set(dropped_keys).isdisjoint(set(processed_hskp_dataset.data_vars.keys()))
    # Check that the dataset has the correct attributes, coordinates, and dimensions
    assert processed_hskp_dataset.attrs == dataset_attrs
    assert processed_hskp_dataset.coords.keys() == dataset_coords_dims

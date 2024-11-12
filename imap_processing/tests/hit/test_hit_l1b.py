import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.hit.l1a import hit_l1a
from imap_processing.hit.l1b import hit_l1b


@pytest.fixture(scope="module")
def packet_filepath():
    """Set path to test data file"""
    return (
        imap_module_directory / "tests/hit/test_data/imap_hit_l0_raw_20100105_v001.pkts"
    )


@pytest.fixture()
def dependencies(packet_filepath):
    """Get dependencies for L1B processing"""
    # Create dictionary of dependencies and add CCSDS packet file
    data_dict = {"imap_hit_l0_raw": packet_filepath}
    # Add L1A datasets
    l1a_datasets = hit_l1a.hit_l1a(packet_filepath, "001")
    for dataset in l1a_datasets:
        data_dict[dataset.attrs["Logical_source"]] = dataset
    return data_dict


@pytest.fixture()
def hk_dataset(dependencies):
    """Get the housekeeping dataset"""
    datasets = hit_l1b.hit_l1b(dependencies, "001")
    for dataset in datasets:
        if dataset.attrs["Logical_source"] == "imap_hit_l1b_hk":
            return dataset


def test_hit_l1b_hk_dataset_variables(hk_dataset):
    """Test the variables in the housekeeping dataset"""
    # Define the keys that should have dropped from the housekeeping dataset
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
    # Define the keys that should be present in the housekeeping dataset
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
    # Check that the dataset has the correct variables
    assert valid_keys == set(hk_dataset.data_vars.keys())
    assert set(dropped_keys).isdisjoint(set(hk_dataset.data_vars.keys()))


def test_hit_l1b_hk_dataset_attributes(hk_dataset):
    """Test the attributes, dims, and coords in the housekeeping dataset"""
    # TODO consider removing this test since it may be hard to upkeep if
    #  attributes change
    # Define the housekeeping dataset attributes
    dataset_attrs = {
        "Acknowledgement": "Please acknowledge the IMAP Mission Principal "
        "Investigator, Prof. David J. McComas of Princeton "
        "University.\n",
        "Data_level": "1B",
        "Data_type": "L1B_HK>Level-1B Housekeeping",
        "Data_version": "001",
        "Descriptor": "HIT>IMAP High-energy Ion Telescope",
        "Discipline": "Solar Physics>Heliospheric Physics",
        "File_naming_convention": "source_descriptor_datatype_yyyyMMdd_vNNN",
        "HTTP_LINK": "https://imap.princeton.edu/",
        "Instrument_type": "Particles (space)",
        "LINK_TITLE": "IMAP The Interstellar Mapping and Acceleration Probe",
        "Logical_file_id": None,
        "Logical_source": "imap_hit_l1b_hk",
        "Logical_source_description": "IMAP Mission HIT Instrument Level-1B "
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

    # Check that the dataset has the correct attributes, coordinates, and dimensions
    assert hk_dataset.attrs == dataset_attrs
    assert hk_dataset.coords.keys() == dataset_coords_dims


def test_hit_l1b(dependencies):
    """Test creating L1B CDF files

    Creates a list of xarray datasets for each L1B product

    Parameters
    ----------
    dependencies : dict
        Dictionary of L1A datasets and CCSDS packet file path
    """
    # TODO: update assertions after science data processing is completed
    datasets = hit_l1b.hit_l1b(dependencies, "001")

    assert len(datasets) == 1
    assert isinstance(datasets[0], xr.Dataset)
    assert datasets[0].attrs["Logical_source"] == "imap_hit_l1b_hk"

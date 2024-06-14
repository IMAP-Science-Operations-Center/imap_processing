from pathlib import Path

import pytest

from imap_processing.cdf.cdf_attribute_manager import CdfAttributeManager


def test_default_attr_schema():
    """
    Test function that covers:
        _load_default_global_attr_schema
        _load_default_variable_attr_schema
    """

    # Initialize CdfAttributeManager object which loads in default schema
    cdf_manager = CdfAttributeManager(Path(__file__).parent.parent / "config")

    # Default global tests
    # Check false case
    assert cdf_manager.global_attribute_schema["DOI"]["required"] is False

    # Check true case
    assert cdf_manager.global_attribute_schema["Data_level"]["required"] is True

    # Default variable tests
    # Check false case
    assert cdf_manager.variable_attribute_schema['attribute_key']["ABSOLUTE_ERROR"]["required"] is False

    # Check true case
    assert cdf_manager.variable_attribute_schema['attribute_key']["RESOLUTION"]["required"] is True


# @pytest.mark.xfail(reason="Missing IMAP specific global schema")


def test_global_attribute():
    """
    Test function that covers:
        load_global_attributes
        get_global_attributes
    """

    # Initialize CdfAttributeManager object which loads in default global variables and schema
    cdf_manager = CdfAttributeManager(Path(__file__).parent.parent / "config")

    # Test that default information was loaded in from "imap_default_global_cdf_attrs.yaml"
    assert cdf_manager.global_attributes["Project"] == "STP>Solar-Terrestrial Physics"
    assert cdf_manager.global_attributes["Source_name"] == "IMAP>Interstellar Mapping and Acceleration Probe"
    assert cdf_manager.global_attributes["Discipline"] == "Solar Physics>Heliospheric Physics"
    assert cdf_manager.global_attributes["Mission_group"] == "IMAP>Interstellar Mapping and Acceleration Probe"
    assert cdf_manager.global_attributes["PI_name"] == "Dr. David J. McComas"
    assert cdf_manager.global_attributes[
               "PI_affiliation"] == "Princeton Plasma Physics Laboratory, 100 Stellarator Road, Princeton, NJ 08540"
    assert cdf_manager.global_attributes["File_naming_convention"] == "source_descriptor_datatype_yyyyMMdd_vNNN"
    assert cdf_manager.global_attributes["Data_version"] == 1

    # Load in different data, test what was carried over
    cdf_manager.load_global_attributes("imap_default_global_test_cdf_attrs.yaml")
    assert cdf_manager.global_attributes["Project"] == "STP>Solar-Terrestrial Physics"
    assert cdf_manager.global_attributes["Source_name"] == "IMAP>Interstellar Mapping and Acceleration Probe"
    assert cdf_manager.global_attributes["Discipline"] == "Solar Physics>Heliospheric Physics"
    assert cdf_manager.global_attributes["Mission_group"] == "Dysfunctional Cats"
    assert cdf_manager.global_attributes["PI_name"] == "Ana Manica"
    assert cdf_manager.global_attributes["PI_affiliation"] == "LASP, CU"
    assert cdf_manager.global_attributes["File_naming_convention"] == "source_descriptor_datatype_yyyyMMdd_vNNN"
    assert cdf_manager.global_attributes["Data_version"] == 2
    assert cdf_manager.global_attributes["Test_attribute"] == "test"

    # Expected failure: file_naming_convention is not in schema.
    # for attr in cdf_manager.global_attributes.keys():
    #     assert attr in cdf_manager.global_attribute_schema.keys()

    # Load additional global attributes
    cdf_manager.load_global_attributes("imap_test_global.yaml")

    # Creating dictionary for loaded information
    mag_test_global_attrs = cdf_manager.get_global_attributes("imap_test_T1_test")

    assert mag_test_global_attrs["Mission_group"] == "Dysfunctional Cats"

    # Testing if statement
    assert mag_test_global_attrs["Descriptor"] == "TEST>Testinstrument"
    assert mag_test_global_attrs["Logical_source"] == "imap_test_T1_test"
    # Testing non required data from schema
    assert mag_test_global_attrs["Data_type"] == "T1_test-one>Test-1 test one"
    # TODO: write call that tests "Bad_name" attribute

    # Testing first elif statement
    assert mag_test_global_attrs["Project"] == cdf_manager.global_attributes["Project"]
    assert mag_test_global_attrs["Source_name"] == cdf_manager.global_attributes["Source_name"]
        # TODO: check with maxine that the below code SHOULD cause an error because "Project" is not in self.global_attributes[inst_id]
        # assert mag_test_global_attrs["Project"] == cdf_manager.global_attributes["imap_test_T1_test"]["Project"]
    assert mag_test_global_attrs["Data_type"] == cdf_manager.global_attributes["imap_test_T1_test"]["Data_type"]

    for attr_name in cdf_manager.global_attributes["imap_test_T1_test"].keys():
        assert attr_name in mag_test_global_attrs.keys()

    # Testing that required schema keys are in get_global_attributes
    for attr_name in cdf_manager.global_attribute_schema.keys():
        required_schema = cdf_manager.global_attribute_schema[attr_name]["required"]
        if required_schema is True:
            assert attr_name in mag_test_global_attrs.keys()


        # BUT not everything in mag_test_global_attrs will be in cdf_manager.global_attributes["imap_test_T1_test"]

    # Testing second elif statement
    # Should throw error



def test_variable_attribute():
    cdf_manager = CdfAttributeManager(Path(__file__).parent.parent / "config")
    cdf_manager.load_global_attributes("imap_default_global_cdf_attrs.yaml")
    cdf_manager.load_variable_attributes("imap_mag_l1a_variable_attrs.yaml")

    # All variables required to have:
    expected_attributes = [
        "DISPLAY_TYPE",
        "FILLVAL",
        "FORMAT",
        "VALIDMIN",
        "VALIDMAX",
        "VAR_TYPE",
    ]

    for variable_attrs in cdf_manager.variable_attributes.values():
        for attr in expected_attributes:
            assert attr in variable_attrs.keys()

    # TODO: Call, and test get_variable_attributes

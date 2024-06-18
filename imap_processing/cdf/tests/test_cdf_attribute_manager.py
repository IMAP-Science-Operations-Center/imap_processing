from pathlib import Path

# import pytest
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
    assert cdf_manager.global_attribute_schema["DOI"]["required"] is False
    assert cdf_manager.global_attribute_schema["Data_level"]["required"] is True

    # Default variable tests
    assert (
        cdf_manager.variable_attribute_schema["attribute_key"]["ABSOLUTE_ERROR"][
            "required"
        ]
        is False
    )
    assert (
        cdf_manager.variable_attribute_schema["attribute_key"]["RESOLUTION"]["required"]
        is True
    )


# @pytest.mark.xfail(reason="Missing IMAP specific global schema")


def test_global_attribute():
    """
    Test function that covers:
        load_global_attributes
    """

    # Initialize CdfAttributeManager object which loads in default info
    cdf_manager = CdfAttributeManager(Path(__file__).parent.parent / "config")
    cdf_manager.load_global_attributes("imap_default_global_cdf_attrs.yaml")

    # Testing information has been loaded in
    assert cdf_manager.global_attributes["Project"] == "STP>Solar-Terrestrial Physics"
    assert (
        cdf_manager.global_attributes["Source_name"]
        == "IMAP>Interstellar Mapping and Acceleration Probe"
    )
    assert (
        cdf_manager.global_attributes["Discipline"]
        == "Solar Physics>Heliospheric Physics"
    )
    assert (
        cdf_manager.global_attributes["Mission_group"]
        == "IMAP>Interstellar Mapping and Acceleration Probe"
    )
    assert cdf_manager.global_attributes["PI_name"] == "Dr. David J. McComas"
    assert (
        cdf_manager.global_attributes["PI_affiliation"]
        == "Princeton Plasma Physics Laboratory, 100 Stellarator Road, "
        "Princeton, NJ 08540"
    )
    assert (
        cdf_manager.global_attributes["File_naming_convention"]
        == "source_descriptor_datatype_yyyyMMdd_vNNN"
    )
    # TODO: write the below test so it fails (it should fail)
    # assert cdf_manager.global_attributes["DOI"] == "test"

    # Load in different data
    cdf_manager.source_dir = Path(__file__).parent.parent / "tests"
    cdf_manager.load_global_attributes("imap_default_global_test_cdf_attrs.yaml")

    # Testing attributes carried over
    assert (
        cdf_manager.global_attributes["File_naming_convention"]
        == "source_descriptor_datatype_yyyyMMdd_vNNN"
    )
    assert (
        cdf_manager.global_attributes["Discipline"]
        == "Solar Physics>Heliospheric Physics"
    )

    # Testing attributes newly loaded
    assert cdf_manager.global_attributes["Project"] == "STP>Solar-Terrestrial Physics"
    assert (
        cdf_manager.global_attributes["Source_name"]
        == "IMAP>Interstellar Mapping and Acceleration Probe"
    )
    assert cdf_manager.global_attributes["Mission_group"] == "Dysfunctional Cats"
    assert cdf_manager.global_attributes["PI_name"] == "Ana Manica"
    assert cdf_manager.global_attributes["PI_affiliation"] == "LASP, CU"
    assert cdf_manager.global_attributes["Data_version"] == 2
    assert cdf_manager.global_attributes["DOI"] == "test"

    # Testing that everything loaded into the global attrs is present in
    #   the global attrs schema
    for attr in cdf_manager.global_attributes.keys():
        assert attr in cdf_manager.global_attribute_schema.keys()


def test_get_global_attributes():
    """
    Test function that covers:
        get_global_attributes
    """
    # Initialize CdfAttributeManager object which loads in default info
    cdf_manager = CdfAttributeManager(Path(__file__).parent.parent / "config")

    # Change filepath to load test global attributes
    cdf_manager.source_dir = Path(__file__).parent.parent / "tests"
    cdf_manager.load_global_attributes("imap_default_global_test_cdf_attrs.yaml")
    cdf_manager.load_global_attributes("imap_test_global.yaml")

    # Loading in instrument specific attributes
    test_get_global_attrs = cdf_manager.get_global_attributes("imap_test_T1_test")

    # Testing information previously loaded into global attributes (first if statement)
    assert test_get_global_attrs["Project"] == "STP>Solar-Terrestrial Physics"
    assert (
        test_get_global_attrs["Source_name"]
        == "IMAP>Interstellar Mapping and Acceleration Probe"
    )
    assert test_get_global_attrs["Mission_group"] == "Dysfunctional Cats"
    # Testing instrument specific global attributes (first elif statement)
    assert test_get_global_attrs["Descriptor"] == "TEST>Testinstrument"
    assert test_get_global_attrs["Data_type"] == "T1_test-one>Test-1 test one"
    assert test_get_global_attrs["Logical_source"] == "imap_test_T1_test"
    assert (
        test_get_global_attrs["Logical_source_description"]
        == "IMAP Mission TEST one document Level-T1."
    )
    # TODO: write call that tests "Bad_name" attribute
    # TODO: Testing second elif statement
    #   How do?

    # Load in more data using get_global_attributes
    test_get_global_attrs_2 = cdf_manager.get_global_attributes("imap_test_T2_test")
    # Testing information previously loaded into global attributes (first if statement)
    assert test_get_global_attrs_2["Project"] == "STP>Solar-Terrestrial Physics"
    # Testing first elif statement
    assert test_get_global_attrs_2["Descriptor"] == "TEST>Testinstrument"
    # "Data_type" not required according to default schema
    assert test_get_global_attrs_2["Data_type"] == "T2_test-two>Test-2 test two"
    # TODO: Testing second elif statement

    # Testing how instrument_id operates
    assert test_get_global_attrs["Project"] == cdf_manager.global_attributes["Project"]
    assert (
        test_get_global_attrs["Source_name"]
        == cdf_manager.global_attributes["Source_name"]
    )
    assert (
        test_get_global_attrs["Data_type"]
        == cdf_manager.global_attributes["imap_test_T1_test"]["Data_type"]
    )
    assert (
        cdf_manager.global_attributes["imap_test_T1_test"]["Logical_source"]
        == "imap_test_T1_test"
    )
    # TODO: The following test should throw an error
    #   assert cdf_manager.global_attributes["imap_test_T1_test"]["Project"]

    # Trying to update a default global using get_global_attributes does not work.
    # For example, thing about DOI event.

    # Testing that required schema keys are in get_global_attributes
    for attr_name in cdf_manager.global_attribute_schema.keys():
        required_schema = cdf_manager.global_attribute_schema[attr_name]["required"]
        if required_schema is True:
            assert attr_name in test_get_global_attrs.keys()


def test_variable_attribute():
    """
    Test function that covers:
        load_variable_attributes
        get_variable_attributes
    """

    # Creating CdfAttributeManager object, loading in default data
    cdf_manager = CdfAttributeManager(Path(__file__).parent.parent / "config")
    cdf_manager.source_dir = Path(__file__).parent.parent / "tests"
    cdf_manager.load_global_attributes("imap_default_global_test_cdf_attrs.yaml")
    # Loading in test data
    cdf_manager.load_variable_attributes("imap_test_variable.yaml")

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


def test_get_variable_attributes():
    # Creating CdfAttributeManager object, loading in default data
    cdf_manager = CdfAttributeManager(Path(__file__).parent.parent / "config")

    # Change filepath to load test global attributes
    cdf_manager.source_dir = Path(__file__).parent.parent / "tests"
    cdf_manager.load_global_attributes("imap_default_global_test_cdf_attrs.yaml")
    cdf_manager.load_variable_attributes("imap_test_variable.yaml")

    # Loading in instrument specific attributes
    imap_test_variable = cdf_manager.get_variable_attributes("test_field_1")

    # Make sure all expected attributes are present
    for variable_attrs in cdf_manager.variable_attribute_schema.keys():
        required_var_attributes = cdf_manager.variable_attribute_schema[variable_attrs]
        if required_var_attributes is True:
            assert variable_attrs in imap_test_variable.keys()

    # Calling required attributes
    assert imap_test_variable["CATDESC"] == "test time"
    assert imap_test_variable["DEPEND_0"] == "test_depend"
    assert imap_test_variable["DISPLAY_TYPE"] == "test_display_type"
    assert imap_test_variable["FIELDNAM"] == "test_field_1"
    assert imap_test_variable["FILLVAL"] == -10
    assert imap_test_variable["FORMAT"] == "I1"
    assert imap_test_variable["LABLAXIS"] == "test_labaxis"
    assert imap_test_variable["UNITS"] == "test_units"
    assert imap_test_variable["VALIDMIN"] == 0
    assert imap_test_variable["VALIDMAX"] == 10
    assert imap_test_variable["VAR_TYPE"] == "test_variable_type"

    # Calling to non required attributes
    assert imap_test_variable["NOT_REQUIRED"] == "test_not_required"

    # Calling attribute name that does not exist
    # TODO: should throw error
    #   assert imap_test_variable["DOES_NOT_EXIST"] == "test time"

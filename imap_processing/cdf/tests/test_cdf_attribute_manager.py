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
    # The following test should fail because "DOI" is not an attribute in
    #   imap_default_global_cdf_attrs.yaml
    with pytest.raises(KeyError):
        assert cdf_manager.global_attributes["DOI"] == "test"

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
    # Not given, and not required information
    assert test_get_global_attrs["Data_level"] is None
    with pytest.raises(KeyError):
        assert test_get_global_attrs["bad_name"] == "false info"

    # Testing second elif statement
    test_error_elif = cdf_manager.get_global_attributes("imap_test_T3_test")
    with pytest.raises(KeyError):
        assert test_error_elif["Data_type"] == "Does Not Exist"

    # Load in more data using get_global_attributes
    test_get_global_attrs_2 = cdf_manager.get_global_attributes("imap_test_T2_test")
    # Testing information previously loaded into global attributes (first if statement)
    assert test_get_global_attrs_2["Project"] == "STP>Solar-Terrestrial Physics"
    # Testing first elif statement
    assert test_get_global_attrs_2["Descriptor"] == "TEST>Testinstrument"
    # "Data_type" not required according to default schema
    assert test_get_global_attrs_2["Data_type"] == "T2_test-two>Test-2 test two"

    # Testing that required schema keys are in get_global_attributes
    for attr_name in cdf_manager.global_attribute_schema.keys():
        required_schema = cdf_manager.global_attribute_schema[attr_name]["required"]
        if required_schema is True:
            assert attr_name in test_get_global_attrs.keys()


def test_instrument_id_format():
    # Initialize CdfAttributeManager object which loads in default info
    cdf_manager = CdfAttributeManager(Path(__file__).parent.parent / "config")

    # Change filepath to load test global attributes
    cdf_manager.source_dir = Path(__file__).parent.parent / "tests"
    cdf_manager.load_global_attributes("imap_default_global_test_cdf_attrs.yaml")
    cdf_manager.load_global_attributes("imap_test_global.yaml")

    # Loading in instrument specific attributes
    test_get_global_attrs = cdf_manager.get_global_attributes("imap_test_T1_test")

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
    with pytest.raises(KeyError):
        assert cdf_manager.global_attributes["imap_test_T1_test"]["Project"]


def test_add_global_attribute():
    # Initialize CdfAttributeManager object which loads in default info
    cdf_manager = CdfAttributeManager(Path(__file__).parent.parent / "config")

    # Change filepath to load test global attributes
    cdf_manager.source_dir = Path(__file__).parent.parent / "tests"
    cdf_manager.load_global_attributes("imap_test_global.yaml")

    # Changing a dynamic global variable
    cdf_manager.add_global_attribute("Project", "Test Project")
    test_get_global_attrs = cdf_manager.get_global_attributes("imap_test_T1_test")
    assert cdf_manager.global_attributes["Project"] == "Test Project"
    assert test_get_global_attrs["Project"] == "Test Project"

    # Testing adding required global attribute
    cdf_manager.global_attributes.__delitem__("Source_name")
    # Reloading get_global_attributes to pick up deleted Source_name
    test_get_global_attrs = cdf_manager.get_global_attributes("imap_test_T1_test")
    with pytest.raises(KeyError):
        assert cdf_manager.global_attributes["Source_name"]
    assert test_get_global_attrs["Source_name"] is None

    # Adding deleted global attribute
    cdf_manager.add_global_attribute("Source_name", "anas_source")
    assert cdf_manager.global_attributes["Source_name"] == "anas_source"
    # Reloading get_global_attributes to pick up added Source_name
    test_get_global_attrs = cdf_manager.get_global_attributes("imap_test_T1_test")
    assert test_get_global_attrs["Source_name"] == "anas_source"

    # Testing instrument specific attribute
    cdf_manager.global_attributes["imap_test_T1_test"].__delitem__("Logical_source")
    # Reloading get_global_attributes to pick up deleted Source_name
    test_get_global_attrs = cdf_manager.get_global_attributes("imap_test_T1_test")
    with pytest.raises(KeyError):
        assert (
            cdf_manager.global_attributes["imap_test_T1_test"]["Logical_source"]
        )
    assert test_get_global_attrs["Logical_source"] is None


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
        "CATDESC",
        "DEPEND_0",
        "DISPLAY_TYPE",
        "FIELDNAM",
        "FILLVAL",
        "FORMAT",
        "LABLAXIS",
        "UNITS",
        "VALIDMIN",
        "VALIDMAX",
        "VAR_TYPE",
    ]

    # Assuring all required attributes are loaded in
    for attr_name in cdf_manager.variable_attribute_schema["attribute_key"]:
        attribute = cdf_manager.variable_attribute_schema["attribute_key"][attr_name]
        if attribute["required"] is True:
            for exp_attr in expected_attributes:
                assert (
                    exp_attr in cdf_manager.variable_attribute_schema["attribute_key"]
                )

    # Testing specific attributes
    assert (
        cdf_manager.variable_attributes["default_attrs"]["DEPEND_0"]
        == cdf_manager.variable_attributes["default_attrs"]["DEPEND_0"]
    )
    assert cdf_manager.variable_attributes["default_attrs"]["FILLVAL"] == -10
    assert cdf_manager.variable_attributes["test_field_1"]["DEPEND_0"] == "test_depend"
    assert (
        cdf_manager.variable_attributes["default_attrs"]["VAR_TYPE"] == "test_var_type"
    )
    with pytest.raises(KeyError):
        assert cdf_manager.variable_attributes["default_attrs"]["CATDESC"] == "test"


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

    # Calling default attributes
    assert imap_test_variable["DEPEND_0"] == "test_depend"
    assert imap_test_variable["DISPLAY_TYPE"] == "test_display_type"
    assert imap_test_variable["FILLVAL"] == -10
    assert imap_test_variable["FORMAT"] == "I1"
    assert imap_test_variable["VALIDMIN"] == 0
    assert imap_test_variable["VALIDMAX"] == 10
    assert imap_test_variable["VAR_TYPE"] == "test_variable_type"

    # Calling required attributes
    assert imap_test_variable["CATDESC"] == "test time"
    assert imap_test_variable["FIELDNAM"] == "test_field_1"
    assert imap_test_variable["LABLAXIS"] == "test_labaxis"
    assert imap_test_variable["UNITS"] == "test_units"
    assert imap_test_variable["VAR_TYPE"] == "test_variable_type"
    assert imap_test_variable["SCALETYP"] == "test_scaletyp"
    assert imap_test_variable["MONOTON"] == "test_monoton"
    assert imap_test_variable["TIME_BASE"] == 10
    assert imap_test_variable["TIME_SCALE"] == "test_time_scale"
    assert imap_test_variable["REFERENCE_POSITION"] == "test_reference_position"

    # Calling to non required attributes
    assert imap_test_variable["NOT_REQUIRED"] == "test_not_required"

    # Calling attribute name that does not exist
    with pytest.raises(KeyError):
        assert imap_test_variable["DOES_NOT_EXIST"] == "test time"

    # Load in different data, test again
    imap_test_variable_2 = cdf_manager.get_variable_attributes("test_field_2")
    # Calling default attributes
    assert imap_test_variable_2["DEPEND_0"] == "test_depend"
    assert imap_test_variable_2["DISPLAY_TYPE"] == "test_display_type"
    assert imap_test_variable_2["FILLVAL"] == -10
    assert imap_test_variable_2["FORMAT"] == "I1"
    assert imap_test_variable_2["VALIDMIN"] == 0
    assert imap_test_variable_2["VALIDMAX"] == 10
    assert imap_test_variable_2["VAR_TYPE"] == "test_variable_type_2"

    # Calling required attributes
    assert imap_test_variable_2["CATDESC"] == "test time 2"
    assert imap_test_variable_2["FIELDNAM"] == "test_field_2"
    assert imap_test_variable_2["LABLAXIS"] == "test_labaxis_2"
    assert imap_test_variable_2["UNITS"] == "test_units_2"
    assert imap_test_variable_2["VAR_TYPE"] == "test_variable_type_2"
    assert imap_test_variable_2["SCALETYP"] == "test_scaletyp_2"
    assert imap_test_variable_2["MONOTON"] == "test_monoton_2"
    assert imap_test_variable_2["TIME_BASE"] == 11
    assert imap_test_variable_2["TIME_SCALE"] == "test_time_scale_2"
    assert imap_test_variable_2["REFERENCE_POSITION"] == "test_reference_position_2"

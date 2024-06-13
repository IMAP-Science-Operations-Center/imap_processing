from pathlib import Path

from imap_processing.cdf.cdf_attribute_manager import CdfAttributeManager

# test __init__(self, data_dir: Path): (?)
# test load_global_attributes(self, file_path: str):
# test _load_yaml_data(file_path: str | Path) -> dict: (?)
# test load_variable_attributes(self, file_name: str) -> None:
# test get_variable_attributes(self, variable_name: str) -> dict:

# test _load_default_global_attr_schema(sef) -> dict: (?)
# test _load_default_variable_attr_schema(self) -> dict: (?)
def test_default_attr_schema():
    # TODO: initialize CdfAttributeManager object
    # TODO: assert statement to test default global schema
    # TODO: assert statement to test default variable schema

def test_load_yaml_data():
    # TODO: Check if file is properly loaded in ... ?

# @pytest.mark.xfail(reason="Missing IMAP specific global schema")
def test_global_attribute():
    cdf_manager = CdfAttributeManager(Path(__file__).parent.parent / "config")

    cdf_manager.load_global_attributes("imap_default_global_cdf_attrs.yaml")

    # Expected failure: file_naming_convention is not in schema.
    # for attr in cdf_manager.global_attributes.keys():
    #     assert attr in cdf_manager.global_attribute_schema.keys()

    assert (
        cdf_manager.global_attributes["Mission_group"]
        == "IMAP>Interstellar Mapping and Acceleration Probe"
    )

    # Load additional global attributes
    cdf_manager.load_global_attributes("imap_mag_global_cdf_attrs.yaml")

    assert (
        cdf_manager.global_attributes["Mission_group"]
        == "IMAP>Interstellar Mapping and Acceleration Probe"
    )

    assert cdf_manager.global_attributes["Descriptor"] == "MAG>Magnetometer"

    mag_l1a_global_attrs = cdf_manager.get_global_attributes("imap_mag_l1a_norm-raw")

    assert (
        mag_l1a_global_attrs["Mission_group"]
        == "IMAP>Interstellar Mapping and Acceleration Probe"
    )
    assert mag_l1a_global_attrs["Descriptor"] == "MAG>Magnetometer"
    assert mag_l1a_global_attrs["Logical_source"] == "imap_mag_l1a_norm-raw"

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
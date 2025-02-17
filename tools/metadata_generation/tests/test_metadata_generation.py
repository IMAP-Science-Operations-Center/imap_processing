import pandas as pd
import pytest
import yaml

from ..metadata_generator import (
    get_dataset_attributes,
    get_global_attributes,
    process_file,
)


@pytest.fixture()
def metadata_spreadsheet(tmp_path):
    """Create an excel file for testing."""
    # Create a pandas DataFrame for global attributes
    data = {
        "NAME": ["global_att1", "global_att2"],
        "FIELD_TYPE": ["global_att", "global_att"],
        "CATDESC": ["description1", "description2"],
    }
    df_global = pd.DataFrame(data)

    # Create a pandas DataFrame for science data
    data = {
        "NAME": ["global_att2", "global_att3", "coord1", "var1", "var2"],
        "FIELD_TYPE": [
            "global_att",
            "global_att",
            "coordinate",
            "variable",
            "variable",
        ],
        "CATDESC": [
            "description1",
            "description2",
            "description3",
            "description4",
            "description5",
        ],
        "VAR_TYPE": ["", "", "", "support_data", "data"],
    }
    df_science = pd.DataFrame(data)

    # Create a pandas DataFrame for science data
    data = {
        "RANDOM": ["random"],
        "COLOR": ["test"],
    }
    df_ignore = pd.DataFrame(data)

    # Write the DataFrame to an excel file
    excel_path = tmp_path / "metadata_spreadsheet.xlsx"
    excel_file = pd.ExcelWriter(excel_path, engine="openpyxl")
    df_global.to_excel(excel_file, sheet_name="global_attrs", index=False)
    df_science.to_excel(excel_file, sheet_name="imap_science_data", index=False)
    df_ignore.to_excel(excel_file, sheet_name="ignore", index=False)
    # Write the file to disk
    excel_file.close()

    return excel_path


def test_get_global_attributes(metadata_spreadsheet):
    """Test the get_global_attributes function."""
    # Read the excel file
    sheet = pd.read_excel(
        metadata_spreadsheet, engine="openpyxl", sheet_name="global_attrs"
    )

    # Get the global attributes
    global_attrs = get_global_attributes(sheet)

    # Check the global attributes
    assert global_attrs == {
        "global_att1": "description1",
        "global_att2": "description2",
    }


def test_get_dataset_attributes(metadata_spreadsheet):
    """Test the get_dataset_attributes function."""
    # Read the excel file
    sheet = pd.read_excel(
        metadata_spreadsheet, engine="openpyxl", sheet_name="imap_science_data"
    )

    # Get the global attributes
    file_attrs = get_dataset_attributes(sheet)

    assert file_attrs == {
        "dataset_attrs": {"global_att2": "description1", "global_att3": "description2"},
        "coord1": {"CATDESC": "description3"},
        "var1": {"CATDESC": "description4", "VAR_TYPE": "support_data"},
        "var2": {"CATDESC": "description5", "VAR_TYPE": "data"},
    }


def test_process_file(metadata_spreadsheet, tmp_path):
    """Test the entire process file function."""
    # Get the global attributes

    process_file(metadata_spreadsheet, tmp_path)
    # First check the global_cdf_attrs
    with open(tmp_path / "global_cdf_attrs.yaml") as f:
        output_dict = yaml.safe_load(f)
    assert output_dict == {
        "imap_science_data": {
            "global_att1": "description1",
            "global_att2": "description2",
            "global_att3": "description2",
        }
    }

    # Now check the variable_attrs
    with open(tmp_path / "variable_attrs.yaml") as f:
        output_dict = yaml.safe_load(f)
    print(output_dict)
    assert output_dict == {
        "coord1": {"CATDESC": "description3"},
        "var1": {"CATDESC": "description4", "VAR_TYPE": "support_data"},
        "var2": {"CATDESC": "description5", "VAR_TYPE": "data"},
    }

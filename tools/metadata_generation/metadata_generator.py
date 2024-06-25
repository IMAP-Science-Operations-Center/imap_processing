"""SWAPI metadata generator."""

from pathlib import Path
from typing import Optional, Union

import pandas as pd
import yaml


def get_global_attributes(sheet: pd.DataFrame) -> pd.DataFrame:
    """
    Get the global attributes from a metadata spreadsheet.

    Parameters
    ----------
    sheet : pandas.DataFrame
        A sheet/tab from an Excel file represented as a pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        A dictionary of the global attributes, where the keys are the attribute names
        and the values are the attribute descriptions.
    """
    sheet = sheet.set_index("NAME")
    # subset by global_att field type, then only keep the CATDESC column
    sheet = sheet[sheet["FIELD_TYPE"] == "global_att"][["CATDESC"]]
    return sheet.to_dict()["CATDESC"]


def get_dataset_attributes(
    sheet: pd.DataFrame, global_attrs: Optional[dict] = None
) -> dict:
    """
    Get the dataset attributes from a metadata spreadsheet.

    Parameters
    ----------
    sheet : pandas.DataFrame
        A sheet/tab from an Excel file represented as a pandas DataFrame.
    global_attrs : dict or None
        List of global attributes.

    Returns
    -------
    dict
        A dictionary of the dataset attributes, where the keys are the attribute names
        and the values are the attribute descriptions.
    """
    # Add the global attributes for this specific sheet/dataset
    file_attrs = {"dataset_attrs": get_global_attributes(sheet)}
    # Add the global attributes from the global_attrs sheet (if any)
    file_attrs["dataset_attrs"].update(global_attrs or {})

    # Move onto variable and coordinate metadata
    # Get everything that isn't a global attribute and drop that column
    variables = sheet[sheet["FIELD_TYPE"] != "global_att"].drop(columns=["FIELD_TYPE"])
    variables = variables.set_index("NAME")

    # Iterate through the rows of the sheet, each row being a variable or coordinate
    # Then drop any NaN values and convert the row to a dictionary of key/values like:
    # {"CATDESC": "description", "VAR_TYPE": "support_data", ...}
    for row, data in variables.iterrows():
        file_attrs[row] = data.dropna().to_dict()

    return file_attrs


# Load all sheets
def process_file(excel_path: Union[str, Path], output_folder: Path) -> None:
    """
    Will process the metadata file and output the metadata to a JSON file.

    Parameters
    ----------
    excel_path : str | pathlib.Path
        The path to the metadata Excel file.
    output_folder : pathlib.Path
        The path to the output folder to place the yaml files.
    """
    sheets = pd.read_excel(excel_path, sheet_name=None)

    # Dictionary to store all of the metadata
    metadata_dict = {}
    global_dataset_attrs = get_global_attributes(sheets["global_attrs"])

    for sheet_name in sheets:
        if not sheet_name.startswith("imap_"):
            # Only worry about the imap_* sheets/tabs
            # ignore support data, and codes
            continue
        metadata_dict[sheet_name] = get_dataset_attributes(
            sheets[sheet_name], global_dataset_attrs
        )

    dataset_metadata = {}
    variable_metadata = {}
    for sheet_name in metadata_dict:
        for variable in metadata_dict[sheet_name]:
            if variable == "dataset_attrs":
                dataset_metadata[sheet_name] = metadata_dict[sheet_name][variable]
            elif variable not in variable_metadata:
                variable_metadata[variable] = metadata_dict[sheet_name][variable]
            # else:
            #     # Variable was already in the dictionary, ignoring for now
            #     # TODO: Do we want to ignore, update, warn?
    with open(output_folder / "global_cdf_attrs.yaml", "w") as f_dataset:
        yaml.dump(dataset_metadata, f_dataset)

    # Save it out to a file
    with open(output_folder / "variable_attrs.yaml", "w") as f:
        yaml.dump(variable_metadata, f)


if __name__ == "__main__":
    process_file(
        "tools/metadata_generation/imap_product_definition_swapi.xlsx",
        Path("imap_processing/cdf/config/"),  # output folder
    )
    print("Finished writing SWAPI metadata file.")

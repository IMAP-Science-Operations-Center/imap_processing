"""
Class for creating and managing CDF attrs.

Developed based of HermesDataSchema from HERMES-SOC/hermes_core.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

DEFAULT_GLOBAL_CDF_ATTRS_FILE = "imap_default_global_cdf_attrs.yaml"
DEFAULT_GLOBAL_CDF_ATTRS_SCHEMA_FILE = "default_global_cdf_attrs_schema.yaml"
DEFAULT_VARIABLE_CDF_ATTRS_SCHEMA_FILE = "default_variable_cdf_attrs_schema.yaml"


class CdfAttributeManager:
    """
    Class for creating and managing CDF attributes based out of yaml files.

    This class is based on the HERMES SWxSOC project for managing CDF attributes, but
    is intended to be a flexible and very lightweight way of managing CDF attribute
    creation and validation.

    To use, you can load one or many global and variable attribute files:

    .. code::

        cdf_attr_manager = CdfAttributeManager(data_dir)
        cdf_attr_manager.load_global_attributes("global_attrs.yaml")
        cdf_attr_manager.load_global_attributes("instrument_global_attrs.yaml")
        cdf_attr_manager.load_variable_attributes("variable_attrs.yaml")

    Later files will overwrite earlier files if the same attribute is defined.

    You can then get the global and variable attributes:

    If you provide an instrument_id, it will also add the attributes defined under
    instrument_id. If this is not included, then only the attributes defined in the top
    level of the file are used.

    .. code::

        # Instrument ID is optional for refining the attributes used from the file
        global_attrs = cdf_attr_manager.get_global_attributes(instrument_id)
        variable_attrs = cdf_attr_manager.get_variable_attributes(variable_name)

    The variable and global attributes are validated against the schemas upon calling
    ``get_global_attributes`` and ``get_variable_attributes``.

    Parameters
    ----------
    data_dir : pathlib.Path
        The directory containing the schema and variable files (nominally config/).

    Attributes
    ----------
    source_dir : pathlib.Path
        The directory containing the schema and variable files - nominally config/
    """

    def __init__(self, data_dir: Path):
        """Initialize the CdfAttributeManager and read schemas from data_dir."""
        # TODO: Split up schema source and data source?
        self.source_dir = data_dir

        # TODO: copied from hermes_core. Currently we can use default schema, but
        # We should add some way of extending the schema and remove all the HERMES
        # specific stuff
        # Data Validation, Complaiance,
        self.global_attribute_schema = self._load_default_global_attr_schema()

        # Data Validation and Compliance for Variable Data
        self.variable_attribute_schema = self._load_default_variable_attr_schema()

        # Load Default IMAP Global Attributes
        self._global_attributes = CdfAttributeManager._load_yaml_data(
            self.source_dir / DEFAULT_GLOBAL_CDF_ATTRS_FILE
        )
        self._variable_attributes: dict = {}

    def _load_default_global_attr_schema(self) -> yaml:
        """
        Load the default global schema from the source directory.

        Returns
        -------
        dict
            The dict representing the global schema.
        """
        default_schema_path = (
            self.source_dir / "shared" / DEFAULT_GLOBAL_CDF_ATTRS_SCHEMA_FILE
        )
        # Load the Schema
        return CdfAttributeManager._load_yaml_data(default_schema_path)

    def _load_default_variable_attr_schema(self) -> yaml:
        """
        Load the default variable schema from the source directory.

        Returns
        -------
        dict
            The dict representing the variable schema.
        """
        default_schema_path = (
            self.source_dir / "shared" / DEFAULT_VARIABLE_CDF_ATTRS_SCHEMA_FILE
        )
        # Load the Schema
        return CdfAttributeManager._load_yaml_data(default_schema_path)

    # TODO Change Returning Any from function declared to return "dict[Any, Any]"

    def load_global_attributes(self, file_path: str) -> None:
        """
        Update the global attributes property with the attributes from the file.

        Calling this method multiple times on different files will add all the
        attributes from the files, overwriting existing attributes if they are
        duplicated.

        Parameters
        ----------
        file_path : str
            File path to load, under self.source_dir.
        """
        self._global_attributes.update(
            CdfAttributeManager._load_yaml_data(self.source_dir / file_path)
        )

    def add_global_attribute(self, attribute_name: str, attribute_value: str) -> None:
        """
        Add a single global attribute to the global attributes.

        This is intended only for dynamic global attributes which change per-file, such
        as Data_version. It is not intended to be used for static attributes, which
        should all be included in the YAML files.

        This will overwrite any existing value in attribute_name if it exists. The
        attribute must be in the global schema, or it will not be included as output.

        Parameters
        ----------
        attribute_name : str
            The name of the attribute to add.
        attribute_value : str
            The value of the attribute to add.
        """
        self._global_attributes[attribute_name] = attribute_value

    @staticmethod
    def _load_yaml_data(file_path: str | Path) -> yaml:
        """
        Load a yaml file from the provided path.

        Parameters
        ----------
        file_path : str | Path
            Path to the yaml file to load.

        Returns
        -------
        yaml
            Loaded yaml.
        """
        with open(file_path) as file:
            return yaml.safe_load(file)

    def get_global_attributes(self, instrument_id: str | None = None) -> dict:
        """
        Generate a dictionary global attributes based off the loaded schema and attrs.

        Validates against the global schema to ensure all required variables are
        present. It can also include instrument specific global attributes if
        instrumet_id is set.

        If an instrument_id is provided, the level and instrument specific
        attributes that were previously loaded using add_instrument_global_attrs will
        be included.

        Parameters
        ----------
        instrument_id : str
            The id of the CDF file, used to retrieve instrument and level
            specific global attributes. Suggested value is the logical_source_id.

        Returns
        -------
        output : dict
            The global attribute values created from the input global attribute files
            and schemas.
        """
        output = dict()
        for attr_name, attr_schema in self.global_attribute_schema.items():
            if attr_name in self._global_attributes:
                output[attr_name] = self._global_attributes[attr_name]
            # Retrieve instrument specific global attributes from the variable file
            elif (
                instrument_id is not None
                and attr_name in self._global_attributes[instrument_id]
            ):
                output[attr_name] = self._global_attributes[instrument_id][attr_name]
            elif attr_schema["required"] and attr_name not in self._global_attributes:
                # TODO throw an error
                output[attr_name] = None

        return output

    def load_variable_attributes(self, file_name: str) -> None:
        """
        Add variable attributes for a given filename.

        Parameters
        ----------
        file_name : str
            The name of the file to load from self.source_dir.
        """
        # Add variable attributes from file_name. Each variable name should have the
        # Required subfields as defined in the variable schema.
        raw_var_attrs = CdfAttributeManager._load_yaml_data(self.source_dir / file_name)
        var_attrs = raw_var_attrs.copy()

        self._variable_attributes.update(var_attrs)

    def get_variable_attributes(
        self, variable_name: str, check_schema: bool = True
    ) -> dict:
        """
        Get the attributes for a given variable name.

        It retrieves the variable from previously loaded variable definition files and
        validates against the defined variable schemas.

        Parameters
        ----------
        variable_name : str
            The name of the variable to retrieve attributes for.

        check_schema : bool
            Flag to bypass schema validation.

        Returns
        -------
        dict
            Information containing specific variable attributes
            associated with "variable_name".
        """
        # Case to handle attributes not in schema
        if check_schema is False:
            if variable_name in self._variable_attributes:
                return_dict: dict = self._variable_attributes[variable_name]
                return return_dict
            # TODO: throw an error?
            return {}

        output = dict()
        for attr_name in self.variable_attribute_schema["attribute_key"]:
            # Standard case
            if attr_name in self._variable_attributes[variable_name]:
                output[attr_name] = self._variable_attributes[variable_name][attr_name]
            # Case to handle DEPEND_i schema issues
            elif attr_name == "DEPEND_i":
                # DEFAULT_0 is not required, UNLESS we are dealing with
                # variable_name = epoch
                # Find all the attributes of variable_name that contain "DEPEND"
                variable_depend_attrs = [
                    key
                    for key in self._variable_attributes[variable_name].keys()
                    if "DEPEND" in key
                ]
                # Confirm that each DEPEND_i attribute is unique
                if len(set(variable_depend_attrs)) != len(variable_depend_attrs):
                    logging.warning(
                        f"Found duplicate DEPEND_i attribute in variable "
                        f"{variable_name}: {variable_depend_attrs}"
                    )
                for variable_depend_attr in variable_depend_attrs:
                    output[variable_depend_attr] = self._variable_attributes[
                        variable_name
                    ][variable_depend_attr]
                # TODO: Add more DEPEND_0 variable checks!
            # Case to handle LABL_PTR_i schema issues
            elif attr_name == "LABL_PTR_i":
                # Find all the attributes of variable_name that contain "LABL_PTR"
                variable_labl_attrs = [
                    key
                    for key in self._variable_attributes[variable_name].keys()
                    if "LABL_PTR" in key
                ]
                for variable_labl_attr in variable_labl_attrs:
                    output[variable_labl_attr] = self._variable_attributes[
                        variable_name
                    ][variable_labl_attr]
            # Case to handle REPRESENTATION_i schema issues
            elif attr_name == "REPRESENTATION_i":
                # Find all the attributes of variable_name that contain
                # "REPRESENTATION_i"
                variable_rep_attrs = [
                    key
                    for key in self._variable_attributes[variable_name].keys()
                    if "REPRESENTATION" in key
                ]
                for variable_rep_attr in variable_rep_attrs:
                    output[variable_rep_attr] = self._variable_attributes[
                        variable_name
                    ][variable_rep_attr]
            # Validating required schema
            elif (
                self.variable_attribute_schema["attribute_key"][attr_name]["required"]
                and attr_name not in self._variable_attributes[variable_name]
            ):
                logging.warning(
                    "Required schema '"
                    + attr_name
                    + "' attribute not present for "
                    + variable_name
                )
                output[attr_name] = ""

        return output

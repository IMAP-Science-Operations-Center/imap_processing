"""
Class for creating and managing CDF attrs.

Developed based of HermesDataSchema from HERMES-SOC/hermes_core.
"""

from __future__ import annotations

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

    ```
    cdf_attr_manager = CdfAttributeManager(data_dir)
    cdf_attr_manager.load_global_attributes("global_attrs.yaml")
    cdf_attr_manager.load_global_attributes("instrument_global_attrs.yaml")
    cdf_attr_manager.load_variable_attributes("variable_attrs.yaml")
    ```

    Later files will overwrite earlier files if the same attribute is defined.

    You can then get the global and variable attributes:

    If you provide an instrument_id, it will also add the attributes defined under
    instrument_id. If this is not included, then only the attributes defined in the top
    level of the file are used.

    ```
    # Instrument ID is optional for refining the attributes used from the file
    global_attrs = cdf_attr_manager.get_global_attributes(instrument_id)
    variable_attrs = cdf_attr_manager.get_variable_attributes(variable_name)
    ```

    The variable and global attributes are validated against the schemas upon calling
    `get_global_attributes` and `get_variable_attributes`.

    Parameters
    ----------
    data_dir : Path
        The directory containing the schema and variable files (nominally config/).

    Attributes
    ----------
    source_dir : Path
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
        self._variable_attributes = dict()

    def _load_default_global_attr_schema(self) -> dict:
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

    def _load_default_variable_attr_schema(self) -> dict:
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

    def load_global_attributes(self, file_path: str):
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
    def _load_yaml_data(file_path: str | Path) -> dict:
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
        # required sub fields as defined in the variable schema.
        raw_var_attrs = CdfAttributeManager._load_yaml_data(self.source_dir / file_name)
        var_attrs = raw_var_attrs.copy()

        self._variable_attributes.update(var_attrs)

    # def get_variable_attributes(self, variable_name: str) -> dict:
    #     """
    #     Get the attributes for a given variable name.
    #
    #     It retrieves the variable from previously loaded variable definition files and
    #     validates against the defined variable schemas.
    #
    #     Parameters
    #     ----------
    #     variable_name : str
    #         The name of the variable to retrieve attributes for.
    #
    #     Returns
    #     -------
    #     dict
    #         I have no idea todo check.
    #     """
    #     # TODO: Create a variable attribute schema file, validate here
    #     if variable_name in self._variable_attributes:
    #         return self._variable_attributes[variable_name]
    #     # TODO: throw an error?
    #     return {}

    def get_variable_attributes(self, variable_name: str, check_schema=True) -> dict:
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
        output = dict()
        for attr_name in self.variable_attribute_schema["attribute_key"]:
            if attr_name in self._variable_attributes[variable_name]:
                output[attr_name] = self._variable_attributes[variable_name][attr_name]
            elif attr_name in self._variable_attributes:
                output[attr_name] = self._variable_attributes[attr_name]
            elif attr_name == "DEPEND_i":
                for i in range(10):
                    attr_name_depend = "DEPEND_" + str(i)
                    if attr_name_depend in self._variable_attributes[variable_name]:
                        output[attr_name_depend] = self._variable_attributes[
                            variable_name
                        ][attr_name_depend]
            elif (
                self.variable_attribute_schema["attribute_key"][attr_name]["required"]
                and attr_name not in self._variable_attributes[variable_name]
                and check_schema is True
                and attr_name != "DEPEND_0"
            ):
                # logger.warn()
                output[attr_name] = ""
            elif check_schema is False:
                if variable_name in self._variable_attributes:
                    return self._variable_attributes[variable_name]
                # TODO: throw an error?
                return {}

        return output

"""Class for creating and managing CDF attrs. Developed based of HermesDataSchema from HERMES-SOC/hermes_core."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import yaml
from imap_processing.cdf import const
import numpy as np

DEFAULT_GLOBAL_CDF_ATTRS_FILE = "imap_default_global_cdf_attrs.yaml"
DEFAULT_GLOBAL_CDF_ATTRS_SCHEMA_FILE = "hermes_default_global_cdf_attrs_schema.yaml"
DEFAULT_VARIABLE_CDF_ATTRS_SCHEMA_FILE = "hermes_default_variable_cdf_attrs_schema.yaml"

class CdfAttributeManager:
    def __init__(self, data_dir: Path):

        # TODO: Split up schema source and data source?
        self.source_dir = data_dir

        # TODO: copied from hermes_core. Currently we can use default schema, but
        # We should add some way of extending the schema and remove all the HERMES specific stuff
        # Data Validation, Complaiance, Derived Attributes
        self.global_attribute_schema = self._load_default_global_attr_schema()

        # Data Validation and Compliance for Variable Data
        self.variable_attribute_schema = (
            self._load_default_variable_attr_schema()
        )

        self.cdftypenames = {
            const.CDF_BYTE.value: "CDF_BYTE",
            const.CDF_CHAR.value: "CDF_CHAR",
            const.CDF_INT1.value: "CDF_INT1",
            const.CDF_UCHAR.value: "CDF_UCHAR",
            const.CDF_UINT1.value: "CDF_UINT1",
            const.CDF_INT2.value: "CDF_INT2",
            const.CDF_UINT2.value: "CDF_UINT2",
            const.CDF_INT4.value: "CDF_INT4",
            const.CDF_UINT4.value: "CDF_UINT4",
            const.CDF_INT8.value: "CDF_INT8",
            const.CDF_FLOAT.value: "CDF_FLOAT",
            const.CDF_REAL4.value: "CDF_REAL4",
            const.CDF_DOUBLE.value: "CDF_DOUBLE",
            const.CDF_REAL8.value: "CDF_REAL8",
            const.CDF_EPOCH.value: "CDF_EPOCH",
            const.CDF_EPOCH16.value: "CDF_EPOCH16",
            const.CDF_TIME_TT2000.value: "CDF_TIME_TT2000",
        }
        self.numpytypedict = {
            const.CDF_BYTE.value: np.int8,
            const.CDF_CHAR.value: np.int8,
            const.CDF_INT1.value: np.int8,
            const.CDF_UCHAR.value: np.uint8,
            const.CDF_UINT1.value: np.uint8,
            const.CDF_INT2.value: np.int16,
            const.CDF_UINT2.value: np.uint16,
            const.CDF_INT4.value: np.int32,
            const.CDF_UINT4.value: np.uint32,
            const.CDF_INT8.value: np.int64,
            const.CDF_FLOAT.value: np.float32,
            const.CDF_REAL4.value: np.float32,
            const.CDF_DOUBLE.value: np.float64,
            const.CDF_REAL8.value: np.float64,
            const.CDF_EPOCH.value: np.float64,
            const.CDF_EPOCH16.value: np.dtype((np.float64, 2)),
            const.CDF_TIME_TT2000.value: np.int64,
        }
        self.timetypes = [
            const.CDF_EPOCH.value,
            const.CDF_EPOCH16.value,
            const.CDF_TIME_TT2000.value,
        ]

        # Load Default IMAP Global Attributes
        self.global_attributes = self._load_yaml_data(
            str(self.source_dir / DEFAULT_GLOBAL_CDF_ATTRS_FILE)
        )
        self.variable_attributes = dict()

    def _load_default_global_attr_schema(self) -> dict:
        default_schema_path = str(
            self.source_dir
            / "shared"
            / DEFAULT_GLOBAL_CDF_ATTRS_SCHEMA_FILE
        )
        # Load the Schema
        return self._load_yaml_data(default_schema_path)

    def _load_default_variable_attr_schema(self) -> dict:
        default_schema_path = str(
            self.source_dir
            / "shared"
            / DEFAULT_VARIABLE_CDF_ATTRS_SCHEMA_FILE
        )
        # Load the Schema
        return CdfAttributeManager._load_yaml_data(default_schema_path)

    def load_global_attributes(self, file_path: str):
        """
        Update the global attributes property with the attributes from the file.

        Parameters
        ----------
        file_path: str
            file path to load, under self.source_dir.

        """
        self.global_attributes.update(
            self._load_yaml_data(
                str(self.source_dir / file_path)
            )
        )

    @staticmethod
    def _load_yaml_data(file_path: str | Path) -> dict:
        """
        Load a yaml file from the provided path.

        Parameters
        ----------
        file_path: str | Path
            path to the yaml file to load.

        Returns
        -------
            Loaded yaml.
        """
        with open(file_path, "r") as file:
            return yaml.safe_load(file)

    def global_attribute_template(self, logical_source_id=None) -> OrderedDict:
        """
        Function to generate a template of required global attributes
        that must be set for a valid CDF.

        If a logical_source_id is provided, the level and instrument specific
        attributes that were previously loaded using add_instrument_global_attrs will
        be included.

        Parameters
        ----------
        logical_source_id : str
            The logical source id of the CDF file, used to retrieve instrument and level
            specific global attributes.

        Returns
        -------
        template : `OrderedDict`
            A template for required global attributes that must be provided.
        """
        # TODO: Move this validation into the load step
        template = OrderedDict()
        for attr_name, attr_schema in self.global_attribute_schema.items():
            if (
                attr_schema["required"]
                and not attr_schema["derived"]
                and attr_name not in self.global_attributes
            ):
                # TODO throw an error?
                template[attr_name] = None
            elif attr_name in self.global_attributes:
                template[attr_name] = self.global_attributes[attr_name]
            # Retrive instrument specific attributes from the instrument template
            elif (
                logical_source_id is not None
                and attr_name in self.global_attributes[logical_source_id]
            ):
                template[attr_name] = self.global_attributes[logical_source_id][
                    attr_name
                ]
        return template

    def load_variable_attrs(self, file_name: str) -> None:
        """
        Add variable attributes for a given filename
        Parameters
        ----------
        file_name: str
            The name of the file to load from self.source_dir
        """
        # Add variable attributes from file_name. Each variable name should have the required sub fields as defined in the variable schema.
        raw_var_attrs = self._load_yaml_data(str(self.source_dir / file_name))
        var_attrs = raw_var_attrs.copy()

        self.variable_attributes.update(var_attrs)

    def variable_attribute_template(self, variable_name) -> OrderedDict:
        # TODO: Create a variable attribute schema file, validate here
        if variable_name in self.variable_attributes:
            return self.variable_attributes[variable_name]
        # TODO: throw an error?
        return OrderedDict()

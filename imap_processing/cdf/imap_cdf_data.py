"""Class for creating and managing CDF attrs. Developed based of HermesDataSchema from HERMES-SOC/hermes_core."""

from collections import OrderedDict
from pathlib import Path

from hermes_core.util.schema import HermesDataSchema

DEFAULT_GLOBAL_CDF_ATTRS_FILE = "imap_default_global_cdf_attrs.yaml"


class ImapCdfData(HermesDataSchema):
    def __init__(self):
        super().__init__()

        # Overwrite attributes with IMAP specific attrs
        self.source_dir = Path(__file__).parent / "config"

        # TODO: I think we can just use the schemas they define in
        #  hermes_core/data/, but we can overwrite if needed

        # Load Default IMAP Global Attributes
        self._default_global_attributes = HermesDataSchema._load_yaml_data(
            str(self.source_dir / DEFAULT_GLOBAL_CDF_ATTRS_FILE)
        )
        self.variable_attributes = dict()

    def add_instrument_global_attrs(self, instrument: str):
        # Looks for file named "imap_{instrument}_global_cdf_attrs.yaml"
        self._default_global_attributes.update(
            HermesDataSchema._load_yaml_data(
                str(self.source_dir / f"imap_{instrument}_global_cdf_attrs.yaml")
            )
        )

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
        template = OrderedDict()
        for attr_name, attr_schema in self.global_attribute_schema.items():
            if (
                attr_schema["required"]
                and not attr_schema["derived"]
                and attr_name not in self.default_global_attributes
            ):
                # TODO throw an error?
                template[attr_name] = None
            elif attr_name in self.default_global_attributes:
                template[attr_name] = self.default_global_attributes[attr_name]
            # Retrive instrument specific attributes from the instrument template
            elif (
                logical_source_id is not None
                and attr_name in self.default_global_attributes[logical_source_id]
            ):
                template[attr_name] = self.default_global_attributes[logical_source_id][
                    attr_name
                ]
        return template

    def add_variable_attrs(self, instrument, level):
        # Add variable attributes from file_name. Each variable name should have the required sub fields as defined in the variable schema.
        raw_var_attrs = HermesDataSchema._load_yaml_data(str(self.source_dir / f"imap_{instrument}_{level}_variable_attrs.yaml"))
        var_attrs = raw_var_attrs.copy()

        for var_name, var_value in raw_var_attrs.items():

            if "base" in var_value.keys():
                # TODO something wonky happening here
                # If there is a base attribute, start there and then add the rest
                var_attrs[var_name] = var_attrs[var_value["base"]]
                var_attrs[var_name].update(var_value)
                print(f"Updating variable {var_name} to use base {var_value['base']}")
                var_attrs[var_name].pop("base")
                print(f"output: {var_attrs[var_name]}")

        self.variable_attributes.update(var_attrs)

    def variable_attribute_template(self, variable_name) -> OrderedDict:
        if variable_name in self.variable_attributes:
            return self.variable_attributes[variable_name]
        # TODO: throw an error?
        return None


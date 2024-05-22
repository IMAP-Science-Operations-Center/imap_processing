"""Class for creating and managing CDF attrs. Developed based of HermesDataSchema from HERMES-SOC/hermes_core."""

from collections import OrderedDict
from pathlib import Path

from hermes_core.util.schema import HermesDataSchema

DEFAULT_GLOBAL_CDF_ATTRS_FILE = "imap_default_global_cdf_attrs.yaml"


class IMAPCDFData(HermesDataSchema):
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
            elif (
                logical_source_id is not None
                and attr_name in self.default_global_attributes[logical_source_id]
            ):
                template[attr_name] = self.default_global_attributes[logical_source_id][
                    attr_name
                ]

        return template

    def add_level_global_attrs(self, logical_source_id):
        # Looks for file named "imap_{logical_source_id}_cdf_attrs.yaml"
        self.file_attributes = dict(
            self._default_global_attributes,
            **HermesDataSchema._load_yaml_data(
                str(self.source_dir / f"imap_{logical_source_id}_cdf_attrs.yaml")
            ),
        )

    def add_variable_attrs(self, file_name):
        # Add variable attributes from file_name. Each variable name should have the required sub fields as defined in the variable schema.
        self.variable_attributes.update(
            HermesDataSchema._load_yaml_data(str(self.source_dir / file_name))
        )
        print(self.variable_attributes)

"""
IMAP specific tools for CDF attribute management.

This class should be used in all cases, and should contain any IMAP specific assumptions
or code.
"""

from __future__ import annotations

from pathlib import Path

from sammi.cdf_attribute_manager import CdfAttributeManager


class ImapCdfAttributes(CdfAttributeManager):  # type: ignore [misc]
    """Contains IMAP specific tools and settings for CDF management."""

    def __init__(self) -> None:
        """Set the path to the config directory."""
        self.source_dir = Path(__file__).parent / "config"

        # TODO: Overwriting schema is not working.
        super().__init__(
            variable_schema_layers=[self.source_dir / "imap_variable_schema.yaml"],
            use_defaults=True,
        )

        # Load constants attrs that everyone uses
        self.load_variable_attributes(self.source_dir / "imap_constant_attrs.yaml")
        self.load_global_attributes(
            self.source_dir / "imap_default_global_cdf_attrs.yaml"
        )

    def add_instrument_global_attrs(self, instrument: str) -> None:
        """
        Add instrument specific global attributes.

        Parameters
        ----------
        instrument : str
            Instrument name.
        """
        # Looks for file named "imap_{instrument}_global_cdf_attrs.yaml"
        self.load_global_attributes(
            self.source_dir / f"imap_{instrument}_global_cdf_attrs.yaml"
        )

    def add_instrument_variable_attrs(
        self, instrument: str, level: str | None = None
    ) -> None:
        """
        Add instrument specific variable attributes.

        Parameters
        ----------
        instrument : str
            Instrument name.
        level : str
            Data level.
        """
        if level is None:
            self.load_variable_attributes(
                self.source_dir / f"imap_{instrument}_variable_attrs.yaml"
            )
        else:
            self.load_variable_attributes(
                self.source_dir / f"imap_{instrument}_{level}_variable_attrs.yaml"
            )

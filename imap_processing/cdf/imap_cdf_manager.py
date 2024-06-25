"""
IMAP specific tools for CDF attribute management.

This class should be used in all cases, and should contain any IMAP specific assumptions
or code.
"""

from pathlib import Path

from imap_processing.cdf.cdf_attribute_manager import CdfAttributeManager


class ImapCdfAttributes(CdfAttributeManager):
    """Contains IMAP specific tools and settings for CDF management."""

    def __init__(self) -> None:
        """Set the path to the config directory."""
        super().__init__(Path(__file__).parent / "config")

    def add_instrument_global_attrs(self, instrument: str) -> None:
        """
        Add instrument specific global attributes.

        Parameters
        ----------
        instrument : str
            Instrument name.
        """
        # Looks for file named "imap_{instrument}_global_cdf_attrs.yaml"
        self.load_global_attributes(f"imap_{instrument}_global_cdf_attrs.yaml")

    def add_instrument_variable_attrs(self, instrument: str, level: str) -> None:
        """
        Add instrument specific variable attributes.

        Parameters
        ----------
        instrument : str
            Instrument name.
        level : str
            Data level.
        """
        self.load_variable_attributes(f"imap_{instrument}_{level}_variable_attrs.yaml")

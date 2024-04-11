"""Parse Lo Level 0 data class attributes."""
from dataclasses import fields


def set_attributes(data, packet):
    """Set Lo Level 0 data class attributes using packet data.

    Parameters
    ----------
    packet : dict
        A single Lo L0 packet from space packet parser.

    """
    attributes = [field.name for field in fields(data)]
    # For each item in packet, assign it to the matching attribute in the class.
    for key, item in packet.data.items():
        value = item.derived_value if item.derived_value is not None else item.raw_value
        if "SPARE" in key or "CHKSUM" in key:
            continue
        if key not in attributes:
            raise KeyError(
                f"Did not find matching attribute in {data.__class__} data class"
                f"for {key}"
            )
        setattr(data, key, value)

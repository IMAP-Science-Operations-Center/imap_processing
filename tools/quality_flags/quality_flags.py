"""Bitwise flagging."""

from enum import STRICT, EnumMeta, IntFlag
from functools import reduce
from operator import or_ as _or_
from typing import Optional


class QualityFlag(IntFlag, boundary=STRICT):
    """Subclass of IntFlag; decomposes flag and returns messages."""

    def __new__(cls, value: int, *args: tuple) -> "QualityFlag":
        """
        Create a new instance of a class.

        Parameters
        ----------
        value : int
            An instance of int that includes an accompanying message.
        *args : tuple
            Additional positional arguments.

        Returns
        -------
        obj : QualityFlag
            A new instance of QualityFlag with the specified value.
            This instance behaves like an integer and it enables
            the inclusion of custom messages.
        """
        obj = int.__new__(cls, value)
        obj._value_ = value
        return obj

    def decompose(self) -> tuple:
        """
        Decompose a flag into its individual components.

        Returns
        -------
        members : list
            Flag values that are subsets of `value`.
        not_covered: int
            Is zero if the OR of members recreates `value`.
            Non-zero otherwise if bits are set in `value`
                that do not exist as named values in cls.
        """
        value = self.value
        not_covered = value

        # Creates the "basis" for the quality flag.
        flags_to_check = []

        # Identify the flags to check
        for m in self.__class__.__members__.values():
            if m.name in (x.name for x in self):
                flags_to_check.append((m, m.value))

        members = []
        for member, member_value in flags_to_check:
            if member_value and member_value & value == member_value:
                members.append(member)
                not_covered &= ~member_value
        members.sort(key=lambda m: m._value_, reverse=True)
        return members, not_covered

    @property
    def summary(self) -> tuple:
        """
        Summarize the quality flag value.

        Returns
        -------
        tuple
            A tuple containing:
            - int: The integer value of the quality flag.
            - list: A list of messages associated with the individual
            flag bits that make up the quality flag.
        """
        members, not_covered = self.decompose()
        if not_covered:
            raise ValueError(
                f"{self.__class__.__name__} has value {self.value} but "
                f"that value cannot be created by elements "
                f"of {self.__class__}."
            )

        try:
            return int(self.value), [m.value.message for m in members]
        except Exception as err:
            raise AttributeError(
                "Tried to summarize a quality flag but its values don't "
                "appear to have messages."
            ) from err


class FlagBit(int):
    """Subclass of int captures integer value and message."""

    # Create a new instance of a class.
    def __new__(cls, value: int, message: Optional[str] = None) -> "FlagBit":
        """
        Create a new instance of FlagBit.

        Parameters
        ----------
        value : int
            The integer value for the FlagBit instance.
        message : str, optional
            The accompanying message for the FlagBit instance.

        Returns
        -------
        FlagBit
            A new instance of FlagBit with the specified value and message.
        """
        # Returns a new instance of int with the given value.
        obj = super().__new__(cls, value)
        # Add message to instance object.
        obj.message = message
        return obj

    # Provides way to view the message
    def __str__(self) -> str:
        """
        Return string representation of the FlagBit instance.

        Returns
        -------
        str
            The string representation of the FlagBit instance in the format
            "value: message".
        """
        return f"{super().__str__()}: {self.message}"


def with_all_none(f: EnumMeta) -> EnumMeta:
    """
    Add NONE and ALL pseudo-members to an enumeration class.

    Parameters
    ----------
    f : EnumMeta
        The enumeration class to which the pseudo-members NONE and ALL will be added.

    Returns
    -------
    f: EnumMeta
        The modified enumeration class with the added pseudo-members NONE and ALL.
    """
    f._member_map_["NONE"] = f(FlagBit(0, message="No flags set."))
    f._member_map_["ALL"] = f(reduce(_or_, f))
    return f

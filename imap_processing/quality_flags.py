"""Bitwise flagging."""

from enum import IntFlag


class CommonFlags(IntFlag):
    NONE = 0x0
    INF = 2**0  # bit 0, Infinite value


class QualityFlags(IntFlag):
    """Quality flags."""

    @property
    def name(self) -> str:
        """
        Override the default name property to handle combined flags.

        Returns
        -------
        combined_name : str
            The combined name of the individual flags.
        """
        if self._name_ is not None:
            return self._name_

        members = [member for member in QualityFlags if member & self == member]
        return "|".join(
            str(m).split(".", 1)[-1] for m in members if m != CommonFlags.NONE
        )


class NewFlag1(QualityFlags):
    """New flag."""

    NONE = CommonFlags.NONE
    INF = CommonFlags.INF
    FLAG2 = 2**1  # bit 1


class NewFlag2(QualityFlags):
    """New flag."""

    NONE = CommonFlags.NONE
    INF = CommonFlags.INF
    FLAG3 = 2**1  # bit 1


# Assuming the previous code has been defined as shown

# Example usage of NewFlag1
flag1 = NewFlag1.NONE
print(f"NewFlag1.NONE: {flag1} ({flag1.name})")

flag2 = NewFlag1.INF
print(f"NewFlag1.INF: {flag2} ({flag2.name})")

flag3 = NewFlag1.FLAG2
print(f"NewFlag1.FLAG2: {flag3} ({flag3.name})")

# Example of combining flags in NewFlag1
combined_flags = NewFlag1.INF | NewFlag1.FLAG2
print(f"Combined NewFlag1: {combined_flags} ({combined_flags.name})")

# Example usage of NewFlag2
flag4 = NewFlag2.NONE
print(f"NewFlag2.NONE: {flag4} ({flag4.name})")

flag5 = NewFlag2.INF
print(f"NewFlag2.INF: {flag5} ({flag5.name})")

flag6 = NewFlag2.FLAG3
print(f"NewFlag2.FLAG3: {flag6} ({flag6.name})")

# Example of combining flags in NewFlag2
combined_flags2 = NewFlag2.INF | NewFlag2.FLAG3
print(f"Combined NewFlag2: {combined_flags2} ({combined_flags2.name})")

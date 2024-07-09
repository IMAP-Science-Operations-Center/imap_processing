from enum import IntFlag
from operator import or_ as _or_
from functools import reduce


class QualityFlag(IntFlag):
    def decompose(self):
        value = self.value
        not_covered = value

        flags_to_check = []

        # Identify the flags to check
        for m in self.__class__.__members__.values():
            if m.name in (x.name for x in self):
                flags_to_check.append((m, m.value))

        members = []
        for member, member_value in flags_to_check:
            # First checks if member_value is non-zero
            # Each bit is set to 1 only if both corresponding bits of the operands are also 1.
            if member_value and member_value & value == member_value:
                members.append(member)
                # Remove the contribution of member_value from not_covered.
                not_covered &= ~member_value
        # No Flags Set.
        if not members and value in self.__class__._value2member_map_:
            members.append(self.__class__._value2member_map_[value])
        members.sort(key=lambda m: m._value_, reverse=True)
        return members, not_covered

    @property
    def summary(self):
        """Summarize quality flag value"""
        members, not_covered = self.decompose()
        print(members)
        if not_covered:
            raise ValueError(f"{self.__name__} has value {self.value} but that value cannot be created by elements "
                             f"of {self.__class__}. This should never happen unless a quality flag was declared "
                             f"without using the FrozenFlagMeta metaclass.")

        try:
            return int(self.value), [m.message for m in members]
        except Exception as err:
            raise AttributeError(
                "Tried to summarize a quality flag but its values don't appear to have messages.") from err


class FlagBit(int):
    """Subclass of int to capture both an integer value and an accompanying message"""
    def __new__(cls, value, message=None):
        obj = super().__new__(cls, value)
        obj.message = message
        return obj

    def __str__(self):
        return f"{super().__str__()}: {self.message}"


def with_all_none(f):
    """Add NONE and ALL psuedo-members to f"""
    f._member_map_['NONE'] = f(FlagBit(0, message="No flags set."))
    f._member_map_['ALL'] = f(reduce(_or_, f))
    return f
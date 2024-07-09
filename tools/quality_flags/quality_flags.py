from enum import IntFlag, EnumMeta
from operator import or_ as _or_
from functools import reduce

class QualityFlagMeta(EnumMeta):
    def __new__(metacls, cls, bases, classdict):
        # Create the new enum class
        enum_class = super().__new__(metacls, cls, bases, classdict)
        # Assign the _name_ attribute and message to each member
        for member_name, member in enum_class._member_map_.items():
            if isinstance(member.value, FlagBit):
                member._name_ = member_name
                member.message = member.value.message
        return enum_class

class QualityFlag(IntFlag, metaclass=QualityFlagMeta):
    def __new__(cls, value, *args):
        obj = int.__new__(cls, value)
        obj._value_ = value
        if isinstance(value, FlagBit):
            obj.message = value.message
        else:
            obj.message = None
        return obj

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
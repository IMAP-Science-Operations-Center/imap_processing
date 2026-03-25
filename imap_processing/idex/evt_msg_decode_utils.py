"""Helper functions for decoding event messages."""

import re

# Regex to match spaces where we need to embed params in the event message templates,
# e.g. {p0}, {p1+2}, {p3:dict}, {p4+1|dict}
EMBEDDED_PARAM_RE = re.compile(r"\{p(\d+)(?:\+(\d+))?(?::[\w]+)?(?:\|(\w+))?\}")


def render_event_template(template: str, params: list, dictionaries: dict) -> str:
    """
    Produce an event message string by replacing placeholders with parameter values.

    Example template:
    "Event {p0} occurred with value {p1+2:dictName}"

    This would replace {p0} with the hex value of params[0], and replace {p1+2:dictName}
    with the combined hex value of params[1] and params[2] (treated as big-endian bytes)
    looked up in the dictionary named "dictName" for a human-readable string, or
    rendered as hex if not found in the dictionary.

    Parameters
    ----------
    template : str
        The event message template containing placeholders like {p0}, {p1+2},
         {p3:dict}, {p4+1|dict}.
    params : list
        The list of parameter values to substitute into the template.
    dictionaries : dict
        Mapping of parameter values to human-readable strings for decoding event
        messages.

    Returns
    -------
    str
        The rendered event message with placeholders replaced by parameter values.
    """

    def replace(m: re.Match[str]) -> str:
        """
        Replace a single placeholder match with the corresponding parameter value.

        Parameters
        ----------
        m : re.Match[str]
            A regex match object for a placeholder in the template.

        Returns
        -------
        str
            The string to replace the placeholder with.
        """
        # group one is the parameter index e.g. p0-3
        idx = int(m.group(1))
        # group two is the optional byte length e.g. +2 (so we know how many params
        # to combine)
        n_bytes = int(m.group(2)) if m.group(2) else 1
        # group three is an optional dictionary name to use for decoding this parameter
        dict_name = m.group(3)
        value = 0
        for i in range(n_bytes):
            # combine the next n_bytes params into a single integer value, treating
            # them as big-endian bytes
            value = (value << 8) | (params[idx + i] if idx + i < len(params) else 0)
        # If a dictionary name is provided use it to decode the value, otherwise just
        # render as hex.
        if dict_name:
            resolved = dictionaries.get(dict_name, {}).get(
                value, f"0x{value:0{2 * n_bytes}X}"
            )
            return f"{dict_name}({resolved})"  # wrap with dict name

        return f"{value:02x}"

    # Replace all placeholders in the template using the replace function defined above.
    return EMBEDDED_PARAM_RE.sub(replace, template)

from pathlib import Path

import yaml
from yaml.nodes import MappingNode

CONFIG_DIR = Path(__file__).parents[2] / "cdf" / "config"


def test_cdf_attribute_yaml_keys_are_sorted():
    """Verify CDF metadata attribute mappings stay alphabetized."""
    unsorted_blocks = []

    for path in sorted(CONFIG_DIR.glob("*_attrs.yaml")):
        doc = yaml.compose(path.read_text())
        for top_key_node, top_value_node in doc.value:
            if not isinstance(top_value_node, MappingNode):
                continue

            keys = [key.value for key, _ in top_value_node.value if key.value != "<<"]
            if keys != sorted(keys):
                unsorted_blocks.append(f"{path.name}:{top_key_node.value}: {keys}")

    assert not unsorted_blocks, (
        "CDF metadata attribute keys must be in alphabetical order:\n"
        + "\n".join(unsorted_blocks)
    )

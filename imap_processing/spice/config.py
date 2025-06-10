"""Spice configuration module."""

from pathlib import Path

# Use a module level attribute to store the location of spin and repoint files
_spin_table_paths: list[Path] | None = None
_repoint_table_path: Path | None = None

"""Precomputed-table move generation (Pentobi-style), proven bit-identical
to the reference generator in ``game.py``."""

from alphablokus.games.blokusduo.movegen.runtime import F2MoveGenerator, get_default_generator
from alphablokus.games.blokusduo.movegen.tables import (
    LookupTable,
    MoveTables,
    build_lookup_table,
    build_move_tables,
)

__all__ = [
    "F2MoveGenerator",
    "LookupTable",
    "MoveTables",
    "build_lookup_table",
    "build_move_tables",
    "get_default_generator",
]

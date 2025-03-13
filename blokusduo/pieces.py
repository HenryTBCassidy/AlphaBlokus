import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Dict, List, Tuple, Generator, Any, Union

import numpy as np
from dataclass_wizard import fromdict
from nptyping import NDArray


class Orientation(StrEnum):
    """
    Possible orientations for a Blokus piece.
    
    Each orientation represents a unique transformation that can be applied to a piece:
    - Identity: No transformation
    - Rot90/180/270: Rotations by 90, 180, or 270 degrees
    - Flip: Mirror reflection
    - Flip90/180/270: Mirror reflection followed by rotation
    """
    Identity = "Identity"
    Rot90 = "Rot90"
    Rot180 = "Rot180"
    Rot270 = "Rot270"
    Flip = "Flip"
    Flip90 = "Flip90"
    Flip180 = "Flip180"
    Flip270 = "Flip270"


@dataclass(frozen=True)
class Piece:
    """
    Represents a single Blokus piece with its properties and possible orientations.

    Attributes:
        name: Unique identifier for the piece
        id: Unique numeric identifier
        basis_orientations: Minimal set of transformations needed to represent all possible orientations
                          For example, the "2-Piece" only needs Identity and Rot90 since other
                          transformations would produce equivalent shapes
        fill_values: 2D grid of 1s and 0s representing the piece's shape
    """
    name: str
    id: int
    basis_orientations: List[Orientation]
    fill_values: List[List[int]]  # Will be converted to np.ndarray, current type due to dataclass_wizard limitation

    @property
    def identity(self) -> NDArray:
        """Return the piece in its original orientation."""
        return np.array(self.fill_values)

    @property
    def rot90(self) -> NDArray:
        """Return the piece rotated 90 degrees clockwise."""
        return np.rot90(self.identity)

    @property
    def rot180(self) -> NDArray:
        """Return the piece rotated 180 degrees."""
        return np.rot90(self.identity, 2)

    @property
    def rot270(self) -> NDArray:
        """Return the piece rotated 270 degrees clockwise."""
        return np.rot90(self.identity, 3)

    @property
    def flip(self) -> NDArray:
        """Return the piece flipped horizontally."""
        return np.flip(self.identity, axis=1)

    @property
    def flip90(self) -> NDArray:
        """Return the piece flipped horizontally then rotated 90 degrees clockwise."""
        return np.rot90(self.flip)

    @property
    def flip180(self) -> NDArray:
        """Return the piece flipped horizontally then rotated 180 degrees."""
        return np.rot90(self.flip, 2)

    @property
    def flip270(self) -> NDArray:
        """Return the piece flipped horizontally then rotated 270 degrees clockwise."""
        return np.rot90(self.flip, 3)


class BidirectionalDict(dict):
    """
    A dictionary that maintains bidirectional mappings between keys and values.
    
    When a key-value pair is added, it automatically adds the reverse mapping.
    This ensures a one-to-one correspondence between keys and values.
    """

    def __setitem__(self, key: Any, val: Any) -> None:
        """
        Add both key->value and value->key mappings.

        Args:
            key: The key to add
            val: The value to add
        """
        dict.__setitem__(self, key, val)
        dict.__setitem__(self, val, key)

    def __delitem__(self, key: Any) -> None:
        """
        Remove both key->value and value->key mappings.

        Args:
            key: The key to remove (also removes the corresponding value mapping)
        """
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)


class PieceManager:
    """
    Manages the collection of Blokus pieces and their transformations.
    
    This class handles the conversion between piece IDs, orientations, and their
    actual board representations. It maintains a bidirectional mapping between
    piece-orientation pairs and their numeric identifiers.
    """

    def __init__(self, pieces: List[Piece]) -> None:
        """
        Initialise the piece manager.

        Args:
            pieces: List of all available Blokus pieces
        """
        self.pieces: Dict[int, Piece] = {p.id: p for p in pieces}
        self._piece_orientation_lookup: BidirectionalDict = self.populate_lookup(pieces)

    def get_piece_orientation_array(self, piece_id: int, orientation: Orientation) -> NDArray:
        """
        Get the board representation of a piece in a specific orientation.

        Args:
            piece_id: ID of the piece
            orientation: Desired orientation

        Returns:
            NDArray: 2D array representing the piece in the specified orientation

        Raises:
            ValueError: If the orientation is not valid
        """
        piece = self.pieces[piece_id]
        match orientation:
            case Orientation.Identity: return piece.identity
            case Orientation.Rot90: return piece.rot90
            case Orientation.Rot180: return piece.rot180
            case Orientation.Rot270: return piece.rot270
            case Orientation.Flip: return piece.flip
            case Orientation.Flip90: return piece.flip90
            case Orientation.Flip180: return piece.flip180
            case Orientation.Flip270: return piece.flip270
            case _: raise ValueError(f"Invalid orientation: {orientation}")

    def all_piece_id_basis_orientations(self) -> Generator[Tuple[int, Orientation], None, None]:
        """
        Generate all valid piece-orientation combinations.

        Yields:
            Tuple containing:
                - piece_id: ID of the piece
                - orientation: One of the piece's valid orientations
        """
        for piece_id, piece in self.pieces.items():
            for orientation in piece.basis_orientations:
                yield piece_id, orientation

    def get_piece_orientation(self, piece_orientation_id: int) -> Tuple[int, Orientation]:
        """
        Convert a piece-orientation ID to its corresponding piece ID and orientation.

        Args:
            piece_orientation_id: Combined ID representing a piece in a specific orientation

        Returns:
            Tuple containing:
                - piece_id: ID of the piece
                - orientation: Orientation of the piece
        """
        return self._piece_orientation_lookup[piece_orientation_id]

    def get_piece_orientation_id(self, piece_orientation: Tuple[int, Orientation]) -> int:
        """
        Convert a piece ID and orientation to their combined piece-orientation ID.

        Args:
            piece_orientation: Tuple of (piece_id, orientation)

        Returns:
            int: Combined ID representing the piece-orientation combination
        """
        return self._piece_orientation_lookup[piece_orientation]

    @property
    def num_entries(self) -> int:
        """Number of unique piece-orientation combinations."""
        return len(self._piece_orientation_lookup)

    @staticmethod
    def populate_lookup(pieces: List[Piece]) -> BidirectionalDict:
        """
        Create a bidirectional mapping between piece-orientation pairs and their IDs.

        Args:
            pieces: List of all available pieces

        Returns:
            BidirectionalDict: Mapping between piece-orientation pairs and their IDs
        """
        lookup = BidirectionalDict()
        i = 1  # IDs start at 1
        for p in pieces:
            for o in p.basis_orientations:
                lookup[i] = (p.id, o)
                i += 1
            i += 1

        return lookup


def pieces_loader(filename: Path) -> PieceManager:
    """
    Load piece definitions from a JSON file and create a PieceManager.

    Args:
        filename: Path to the JSON file containing piece definitions

    Returns:
        PieceManager: Manager containing all loaded pieces

    Raises:
        AssertionError: If the piece data is invalid (wrong number of pieces,
                       orientations, or non-unique IDs)
        ValueError: If the piece-orientation lookup table is invalid
    """
    with open(filename, "rb") as f:
        d = json.load(f)
    result = [fromdict(Piece, item) for item in d]

    ids = set(r.id for r in result)
    num_orientations = sum(len(r.basis_orientations) for r in result)

    if not len(result) == 21:
        raise AssertionError(f"Expected 21 pieces, found {len(result)} unique ids for {len(result)} pieces.")

    if not num_orientations == 91:
        raise AssertionError(f"Configured net for 91 basis orientations, found {num_orientations}!")

    if not len(ids) == len(result):
        raise AssertionError(f"Ids are not unique! Found {len(ids)} unique ids for {len(result)} pieces.")

    piece_manager = PieceManager(pieces=result)

    if piece_manager.num_entries != 2 * 91:  # twice 91 since values are put in both ways
        raise ValueError(
            f"Expected {2 * 91} entries in PieceManager, got {piece_manager.num_entries}. "
            f"Error constructing Piece-Orientation <-> id lookup!")

    return piece_manager

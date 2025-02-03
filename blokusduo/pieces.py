import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from dataclass_wizard import fromdict
from nptyping import NDArray


class Orientation(StrEnum):
    """
    Contains the set of possible orientations for a Piece
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
    A class containing information about one of the 21 Blokus Pieces.

    Attributes:
        name (str): Unique name for Piece.
        id (int): Unique id for Piece.
        basis_orientations (list[Piece]): The smallest set of transformations made to a piece to span all possible
                                          moves for the Piece
                                          e.g. The "2-Piece" can either be [[1, 1]] or [[1],
                                                                                        [1]]
                                          i.e. Not be transformed (Identity) or rotated by 90 degrees (Rot90).
                                          No further transformations are needed since rot180 == flip == identity etc.
        fill_values (list[list[int]]): A list of 1 and 0 that is reshaped by length and width into the Piece.shape.

        Additionally, has a method for each value in Orientation(StrEnum) to get the board representation for said
        transformation.
    """
    name: str
    id: int
    basis_orientations: list[Orientation]
    fill_values: list[list[int]]  # Converted into np.array, this dtype is not yet supported by dataclass_wizard

    @property
    def identity(self) -> NDArray: return np.array(self.fill_values)

    @property
    def rot90(self) -> NDArray: return np.rot90(self.identity)

    @property
    def rot180(self) -> NDArray: return np.rot90(self.identity, 2)

    @property
    def rot270(self) -> NDArray: return np.rot90(self.identity, 3)

    @property
    def flip(self) -> NDArray: return np.flip(self.identity, axis=1)

    @property
    def flip90(self) -> NDArray: return np.rot90(self.flip)

    @property
    def flip180(self) -> NDArray: return np.rot90(self.flip, 2)

    @property
    def flip270(self) -> NDArray: return np.rot90(self.flip, 3)


class BidirectionalDict(dict):
    """
    Dictionary which you add a (key, value) pair to, also populates (value, key) into dict.
    Useful for making sure there is a one to one mapping between data
    """

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)
        dict.__setitem__(self, val, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)


class PieceManager:
    """
    Class for holding pieces and helping convert actions into actual piece moves or allowing the net to understand a
    piece.
    Knows how to turn a piece_id and orientation into
    """

    def __init__(self, pieces: list[Piece]):
        self.pieces = {p.id: p for p in pieces}
        self._piece_orientation_lookup = self.populate_lookup(pieces)

    def get_piece_orientation_array(self, piece_id: int, orientation: Orientation) -> NDArray:
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
            case _: raise ValueError(f"Trying to find Orientation {orientation}, could not match value!")

    def all_piece_id_basis_orientations(self):
        """
        Generator returns tuple[piece_id, orientation]
        :return:
        """
        for piece_id, piece in self.pieces.items():
            for orientation in piece.basis_orientations:
                yield piece_id, orientation

    def get_piece_orientation(self, piece_orientation_id: int) -> tuple[int, Orientation]:
        return self._piece_orientation_lookup[piece_orientation_id]

    def get_piece_orientation_id(self, piece_orientation: tuple[int, Orientation]) -> int:
        return self._piece_orientation_lookup[piece_orientation]

    @property
    def num_entries(self) -> int:
        return len(self._piece_orientation_lookup)

    @staticmethod
    def populate_lookup(pieces: list[Piece]) -> BidirectionalDict:
        lookup = BidirectionalDict()
        i = 1  # Ids so they start at 1
        for p in pieces:
            for o in p.basis_orientations:
                lookup[i] = p.id, o
                i += 1
            i += 1

        return lookup


def pieces_loader(filename: Path) -> PieceManager:
    """
    Helper method for loading in the pieces from the target json file
    Runs checks to make sure data is not corrupted
    :param filename: File location of pieces in json format
    :return: PieceManager
    """
    with open(filename, "rb") as f:
        d = json.load(f)
    result = [fromdict(Piece, item) for item in d]

    ids = set([r.id for r in result])
    num_orientations = np.sum([len(r.basis_orientations) for r in result])

    if not len(result) == 21:
        raise AssertionError(f"Expected 21 pieces, found {len(result)} unique ids for {len(result)} pieces.")

    if not num_orientations == 91:
        raise AssertionError(f"Configured net for 91 basis orientations, found {num_orientations}!")

    if not len(ids) == len(result):
        raise AssertionError(f"Ids are not unique! Found {len(ids)} unique ids for {len(result)} pieces.")

    piece_manager = PieceManager(pieces=result)

    if piece_manager.num_entries != 2 * 91:  # twice 91 since values are put in both ways
        raise ValueError(
            f"Expected {2 * 91} entries in PieceManager, got {piece_manager.num_entries}."
            f"Error constructing Piece-Orientation <-> id lookup!")

    return piece_manager

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from dataclass_wizard import fromdict


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


@dataclass
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
    def identity(self) -> np.array: return np.array(self.fill_values)

    @property
    def rot90(self) -> np.array: return np.rot90(self.identity)

    @property
    def rot180(self) -> np.array: return np.rot90(self.identity, 2)

    @property
    def rot270(self) -> np.array: return np.rot90(self.identity, 3)

    @property
    def flip(self) -> np.array: return np.flip(self.identity, axis=1)

    @property
    def flip90(self) -> np.array: return np.rot90(self.flip)

    @property
    def flip180(self) -> np.array: return np.rot90(self.flip, 2)

    @property
    def flip270(self) -> np.array: return np.rot90(self.flip, 3)


def pieces_loader(filename: Path) -> list[Piece]:
    """
    Helper method for loading in the pieces from the target json file
    Runs checks to make sure data is not corrupted
    :param filename: File location of pieces in json format
    :return: list[Piece]
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

    return result

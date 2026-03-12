from games.blokusduo.pieces import BidirectionalDict, PieceManager


def test_encode_decode_roundtrip(piece_manager: PieceManager):
    """Every (piece_id, orientation) pair should survive encode → decode."""
    for piece_id, piece in piece_manager.pieces.items():
        for orientation in piece.basis_orientations:
            pair = (piece_id, orientation)
            encoded_id = piece_manager.get_piece_orientation_id(pair)
            decoded = piece_manager.get_piece_orientation(encoded_id)
            assert decoded == pair, f"Roundtrip failed for {pair}: got {decoded}"


def test_all_ids_unique(piece_manager: PieceManager):
    """No two (piece_id, orientation) pairs should map to the same ID."""
    seen_ids: set[int] = set()
    for piece_id, piece in piece_manager.pieces.items():
        for orientation in piece.basis_orientations:
            pair = (piece_id, orientation)
            encoded_id = piece_manager.get_piece_orientation_id(pair)
            assert encoded_id not in seen_ids, f"Duplicate ID {encoded_id} for {pair}"
            seen_ids.add(encoded_id)


def test_bidirectional_dict():
    """BidirectionalDict should maintain forward and reverse mappings."""
    bd = BidirectionalDict()
    bd["a"] = 1
    bd["b"] = 2

    # Forward lookup
    assert bd["a"] == 1
    assert bd["b"] == 2

    # Reverse lookup
    assert bd[1] == "a"
    assert bd[2] == "b"


def test_bidirectional_dict_delete():
    """Deleting a key should remove both mappings."""
    bd = BidirectionalDict()
    bd["a"] = 1
    del bd["a"]
    assert "a" not in bd
    assert 1 not in bd

from games.blokusduo.pieces import OrientationCodec, PieceManager


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


def test_orientation_ids_contiguous(piece_manager: PieceManager):
    """IDs should be 0-based and contiguous with no gaps."""
    all_ids: list[int] = []
    for piece_id, piece in piece_manager.pieces.items():
        for orientation in piece.basis_orientations:
            all_ids.append(piece_manager.get_piece_orientation_id((piece_id, orientation)))

    all_ids.sort()
    assert all_ids == list(range(91))


def test_orientation_codec_len(piece_manager: PieceManager):
    """OrientationCodec len should equal the number of piece-orientation pairs."""
    assert piece_manager.num_entries == 91


def test_orientation_codec_contains(piece_manager: PieceManager):
    """OrientationCodec should support membership testing for both IDs and pairs."""
    codec = piece_manager._orientation_codec
    for piece_id, piece in piece_manager.pieces.items():
        for orientation in piece.basis_orientations:
            pair = (piece_id, orientation)
            orientation_id = piece_manager.get_piece_orientation_id(pair)
            assert orientation_id in codec
            assert pair in codec

    assert -1 not in codec
    assert 91 not in codec

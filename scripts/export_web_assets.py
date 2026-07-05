"""Export the web-play engine assets: rules blob, pieces JSON, manifest, ONNX net.

The browser engine (``web/src/engine/``) must agree with the Python reference
engine exactly. Rather than reimplementing move generation in TypeScript, we
export the same static geometry tables the JAX kernels are built from
(``movegen.tables.build_move_tables`` — footprint / edge-halo / corner-halo
cell lists per action) as a little-endian binary blob, plus the piece
orientation grids for tray rendering and a manifest describing every artifact.
Parity is then by construction, verified by the fixture battery
(``scripts/generate_web_parity_fixtures.py``).

Usage:
    uv run python scripts/export_web_assets.py --rules-only
    uv run python scripts/export_web_assets.py \
        --config run_configurations/blokus_run3_overnight.json \
        --checkpoint temp/runs/blokus/blokus_run3_overnight/Nets/accepted_82.pth.tar

The ONNX export (checkpoint mode) requires the ``web`` extra:
``uv sync --extra web``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.movegen.tables import NULL_CELL, build_move_tables
from alphablokus.games.blokusduo.pieces import default_pieces_path

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Bumped whenever the board encoding or blob layout changes incompatibly.
ENCODING_VERSION = "blokusduo-44ch-v1"


@dataclass(frozen=True)
class BlobArray:
    """One named array inside the rules blob, as recorded in the manifest."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    num_bytes: int


def _sha256(path: Path) -> str:
    """Hex digest of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_rules_blob(game: BlokusDuoGame, out_path: Path) -> list[BlobArray]:
    """Write the static rules tables as one concatenated little-endian blob.

    Array order (all little-endian, offsets recorded in the returned entries):
    ``piece`` u8[M], ``action_id`` u32[M], ``cells`` u8[M,5],
    ``adj_cells`` u8[M,16], ``attach_cells`` u8[M,16], ``piece_sizes`` u8[22],
    ``start_cells`` i32[2] (flat array-index start cell per player slot).
    """
    tables = build_move_tables(game.piece_manager)

    piece_sizes = np.zeros(22, dtype=np.uint8)
    for piece_id, piece in game.piece_manager.pieces.items():
        piece_sizes[piece_id] = int(piece.identity.sum())

    start_cells = np.empty(2, dtype=np.int32)
    for slot, (row, col) in enumerate((game.white_start, game.black_start)):
        start_cells[slot] = row * game.board_size + col

    arrays: list[tuple[str, NDArray]] = [
        ("piece", tables.piece.astype(np.uint8)),
        ("action_id", tables.action_id.astype(np.uint32)),
        ("cells", tables.cells.astype(np.uint8)),
        ("adj_cells", tables.adj_cells.astype(np.uint8)),
        ("attach_cells", tables.attach_cells.astype(np.uint8)),
        ("piece_sizes", piece_sizes),
        ("start_cells", start_cells),
    ]

    entries: list[BlobArray] = []
    offset = 0
    with open(out_path, "wb") as blob:
        for name, array in arrays:
            little_endian = array.astype(array.dtype.newbyteorder("<"), copy=False)
            raw = little_endian.tobytes(order="C")
            blob.write(raw)
            entries.append(
                BlobArray(
                    name=name,
                    dtype=str(array.dtype),
                    shape=tuple(int(dim) for dim in array.shape),
                    offset=offset,
                    num_bytes=len(raw),
                )
            )
            offset += len(raw)

    logger.info("Wrote rules blob: {} ({} bytes, {} moves)", out_path, offset, tables.num_moves)
    return entries


def _write_pieces_json(game: BlokusDuoGame, out_path: Path) -> None:
    """Write the 91 orientation grids + piece metadata for tray rendering."""
    manager = game.piece_manager
    orientations: list[dict[str, Any]] = []
    for orientation_id in range(manager.num_entries):
        piece_id, orientation = manager.get_piece_orientation(orientation_id)
        grid = manager.get_piece_orientation_array(piece_id, orientation)
        orientations.append(
            {
                "orientationId": orientation_id,
                "pieceId": piece_id,
                "orientation": orientation.value,
                "grid": grid.astype(int).tolist(),
            }
        )
    pieces = [
        {"id": piece_id, "name": piece.name, "size": int(piece.identity.sum())}
        for piece_id, piece in sorted(game.piece_manager.pieces.items())
    ]
    out_path.write_text(json.dumps({"pieces": pieces, "orientations": orientations}, indent=1))
    logger.info("Wrote pieces JSON: {}", out_path)


def _export_onnx(
    config_path: Path,
    checkpoint_path: Path,
    out_path: Path,
    *,
    fp16: bool,
    int8: bool,
) -> dict[str, Any]:
    """Export the torch checkpoint to ONNX; return the manifest ``net`` entry.

    The graph's outputs are exactly the torch forward's: ``log_policy``
    (log-softmax over the 17,837 actions) and ``value`` (tanh scalar), with a
    dynamic batch axis. fp32 is the parity baseline; ``--fp16``/``--int8``
    additionally write quantised variants for smaller downloads.
    """
    import torch

    from alphablokus.config import load_args
    from alphablokus.registry import instantiate_game_and_network

    config = load_args(config_path)
    if config.game != "blokusduo":
        raise ValueError(f"Web export supports blokusduo only, config says {config.game!r}.")

    game, wrapper = instantiate_game_and_network(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    wrapper.nnet.load_state_dict(checkpoint["state_dict"])
    net = wrapper.nnet.to("cpu")
    net.eval()

    board = game.initialise_board()
    example = torch.zeros((1, board.num_channels, *game.get_board_size()), dtype=torch.float32)
    torch.onnx.export(
        net,
        (example,),
        str(out_path),
        input_names=["board"],
        output_names=["log_policy", "value"],
        dynamic_axes={"board": {0: "batch"}, "log_policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
    )
    logger.info("Wrote ONNX net: {} ({:.1f} MB)", out_path, out_path.stat().st_size / 1e6)

    variants: dict[str, str] = {"fp32": out_path.name}
    if fp16:
        variants["fp16"] = _quantise_fp16(out_path).name
    if int8:
        variants["int8"] = _quantise_int8(out_path).name

    num_params = sum(int(p.numel()) for p in net.parameters())
    return {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "numFilters": config.net_config.num_filters,
        "numResidualBlocks": config.net_config.num_residual_blocks,
        "policyHead": config.net_config.policy_head,
        "numParameters": num_params,
        "variants": variants,
    }


def _quantise_fp16(fp32_path: Path) -> Path:
    """Write a float16 copy of the ONNX model next to the fp32 one."""
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(fp32_path))
    model_fp16 = float16.convert_float_to_float16(model)
    out_path = fp32_path.with_name(fp32_path.stem + ".fp16.onnx")
    onnx.save(model_fp16, str(out_path))
    logger.info("Wrote fp16 net: {} ({:.1f} MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


def _quantise_int8(fp32_path: Path) -> Path:
    """Write a dynamically-quantised int8 copy of the ONNX model."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_path = fp32_path.with_name(fp32_path.stem + ".int8.onnx")
    quantize_dynamic(str(fp32_path), str(out_path), weight_type=QuantType.QInt8)
    logger.info("Wrote int8 net: {} ({:.1f} MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


def export_assets(
    out_dir: Path,
    *,
    config_path: Path | None,
    checkpoint_path: Path | None,
    fp16: bool = False,
    int8: bool = False,
) -> Path:
    """Export all web assets into ``out_dir``; return the manifest path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    game = BlokusDuoGame(pieces_config_path=default_pieces_path())

    rules_path = out_dir / "rules.bin"
    blob_entries = _write_rules_blob(game, rules_path)
    pieces_path = out_dir / "pieces.json"
    _write_pieces_json(game, pieces_path)

    manifest: dict[str, Any] = {
        "encodingVersion": ENCODING_VERSION,
        "boardSize": game.board_size,
        "numCells": game.board_size * game.board_size,
        "actionSize": game.get_action_size(),
        "passIndex": game.action_codec.pass_action_index,
        "numOrientations": game.num_orientations,
        "numChannels": game.initialise_board().num_channels,
        "nullCell": NULL_CELL,
        "rules": {
            "path": rules_path.name,
            "sha256": _sha256(rules_path),
            "bytes": rules_path.stat().st_size,
            "arrays": [vars(entry) | {"shape": list(entry.shape)} for entry in blob_entries],
        },
        "pieces": {"path": pieces_path.name, "sha256": _sha256(pieces_path)},
        "net": None,
    }

    if checkpoint_path is not None:
        assert config_path is not None  # argparse enforces the pairing
        model_path = out_dir / "model.onnx"
        net_entry = _export_onnx(config_path, checkpoint_path, model_path, fp16=fp16, int8=int8)
        net_entry["files"] = {
            variant: {"path": name, "sha256": _sha256(out_dir / name), "bytes": (out_dir / name).stat().st_size}
            for variant, name in net_entry.pop("variants").items()
        }
        manifest["net"] = net_entry

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1))
    logger.info("Wrote manifest: {}", manifest_path)
    return manifest_path


def main() -> int:
    """CLI entry point."""
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "web" / "public" / "assets",
        help="Output directory for the exported assets.",
    )
    parser.add_argument("--rules-only", action="store_true", help="Skip the ONNX net export (no checkpoint needed).")
    parser.add_argument("--config", type=Path, default=None, help="Run-config JSON describing the net architecture.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Torch checkpoint (.pth.tar) to export.")
    parser.add_argument("--fp16", action="store_true", help="Also write a float16 ONNX variant.")
    parser.add_argument("--int8", action="store_true", help="Also write a dynamically-quantised int8 ONNX variant.")
    args = parser.parse_args()

    if not args.rules_only and (args.config is None or args.checkpoint is None):
        parser.error("--config and --checkpoint are required unless --rules-only is set.")

    export_assets(
        args.out,
        config_path=None if args.rules_only else args.config,
        checkpoint_path=None if args.rules_only else args.checkpoint,
        fp16=args.fp16,
        int8=args.int8,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

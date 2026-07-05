"""FastAPI wrapper for :class:`PlayService` + static hosting of the frontend.

The download tier: one command serves the same built frontend as the static
site, but ``/api/best-move`` answers with the real torch/MCTS stack at full
strength. Composition (game + net from the run config) happens here through
``registry`` — the service itself is protocol-only.

Run:
    uv sync --extra play
    uv run alphablokus-play --config run_configurations/<run>.json \
        --checkpoint <path/to/checkpoint.pth.tar>
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from alphablokus.config import load_args
from alphablokus.play.service import PlayService
from alphablokus.registry import instantiate_game_and_network


class BestMoveRequest(BaseModel):
    """Move request: the game so far (flat action ids) + a strength level."""

    history: list[int]
    difficulty: str


class BestMoveResponse(BaseModel):
    action: int
    value: float
    legal: list[int]
    sims: int
    elapsedMs: float


class LegalMovesRequest(BaseModel):
    history: list[int]


def create_app(service: PlayService, web_dist: Path | None) -> FastAPI:
    """Build the FastAPI app: engine API + (optionally) the static frontend."""
    app = FastAPI(title="AlphaBlokus play server")
    # The vite dev server (npm run dev) proxies from another localhost port
    # during frontend development; the built frontend is same-origin.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/meta")
    def meta() -> dict:
        return {
            "name": "Local full-strength engine (torch + MCTS)",
            "isFullStrength": True,
            "actionSize": service.action_size,
            "difficulties": [
                {
                    "id": level.id,
                    "label": level.label,
                    "searchPolicy": level.search_policy,
                    "sims": level.sims,
                    "description": level.description,
                }
                for level in service.difficulties
            ],
        }

    @app.post("/api/legal-moves")
    def legal_moves(request: LegalMovesRequest) -> dict:
        return {"legal": service.legal_actions(request.history)}

    @app.post("/api/best-move")
    def best_move(request: BestMoveRequest) -> BestMoveResponse:
        try:
            result = service.best_move(request.history, request.difficulty)
        except KeyError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return BestMoveResponse(
            action=result.action,
            value=result.value,
            legal=result.legal,
            sims=result.sims,
            elapsedMs=result.elapsed_ms,
        )

    if web_dist is not None and web_dist.exists():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="frontend")
    else:
        logger.warning("Frontend build not found at {} — API only. Run `npm run build` in web/.", web_dist)

    return app


def main() -> int:
    """CLI entry point (console script ``alphablokus-play``)."""
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Run-config JSON (game + net architecture).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Torch checkpoint (.pth.tar) to play with.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=8000, help="Port.")
    parser.add_argument(
        "--web-dist",
        type=Path,
        default=repo_root / "web" / "dist",
        help="Built frontend to serve (web/dist).",
    )
    args = parser.parse_args()

    config = load_args(args.config)
    if config.net_config.cuda and not torch.cuda.is_available():
        # Training configs from the GPU box say cuda: true; play anywhere by
        # degrading to CPU (also makes load_checkpoint map to CPU).
        logger.info("CUDA unavailable — running the net on CPU.")
        config = replace(config, net_config=replace(config.net_config, cuda=False))
    game, nnet = instantiate_game_and_network(config)
    # ``load_checkpoint`` joins ``net_directory / filename``; an absolute
    # filename wins the join, so any checkpoint path loads through the protocol.
    nnet.load_checkpoint(str(args.checkpoint.resolve()))
    logger.info("Loaded checkpoint {} for game {!r}", args.checkpoint, config.game)

    service = PlayService(game, nnet, mcts_batch_size=config.mcts_config.mcts_batch_size)
    app = create_app(service, args.web_dist)
    logger.info("Serving on http://{}:{} (full-strength local engine)", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

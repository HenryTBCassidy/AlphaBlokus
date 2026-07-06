"""CLI wrapper for the post-hoc pool BayesElo tournament.

The tournament logic lives in the installable package
(:mod:`alphablokus.evaluation.tournament_run`) so ``cli.main`` can also invoke it
automatically at end-of-run (``RunConfig.tournament.run_at_end``). This module is
just the standalone command-line front end.

Usage::

    uv run python -m scripts.tournament_elo --config run_configurations/blokus_cloud.json
    uv run python -m scripts.tournament_elo --config <cfg> --dry-run   # schedule + game count only

Outputs land in ``<run>/Tournament/`` (ratings parquet + raw W/L/D JSON) — see
:mod:`alphablokus.evaluation.tournament_run` for the format.
"""

from __future__ import annotations

import argparse

from alphablokus.config import load_args
from alphablokus.evaluation.tournament_run import run_tournament

__all__ = ["run_tournament"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the run's RunConfig JSON.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pairing schedule and total game count, then exit without playing.",
    )
    args = parser.parse_args()

    config = load_args(args.config)
    run_tournament(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

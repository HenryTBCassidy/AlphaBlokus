"""Subprocess client for the Pentobi GTP engine (H3).

Drives a ``pentobi-gtp`` process over stdin/stdout using GTP (Go Text Protocol): we
write a command line, flush, and read the response — lines starting with ``=`` (success)
or ``?`` (failure), terminated by a blank line. This is the low-level transport; the
higher-level ``PentobiPlayer`` (H4) wraps it with the move translation (H1).

Interface facts pinned in H2 against the real binary (Pentobi v31, ``--game duo``):
- colour tokens are ``b`` / ``w`` (our White=+1 ↔ ``b``; Black=−1 ↔ ``w`` — see
  ``pentobi_translation`` / the harness plan);
- moves are comma-separated lowercase cells (``h7,g8,...``) or ``pass``;
- ``final_score`` returns ``B+N`` / ``W+N`` (``B`` = our White).

Gotchas handled (per docs/06-INTERFACES.md §2): flush after every write; read to the
**blank-line** terminator (not one line); ``--quiet`` + stderr→DEVNULL so the engine's
logging can't fill a pipe and deadlock; EOF on stdout ⇒ the engine died.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

# Default location of the locally-built binary (H2). Override via $PENTOBI_GTP_PATH.
DEFAULT_PENTOBI_GTP = Path.home() / "code" / "pentobi" / "build" / "pentobi_gtp" / "pentobi-gtp"


def find_pentobi_gtp() -> Path | None:
    """Locate the pentobi-gtp binary: $PENTOBI_GTP_PATH, else the default build path."""
    env = os.environ.get("PENTOBI_GTP_PATH")
    if env:
        p = Path(env).expanduser()
        return p if p.exists() else None
    return DEFAULT_PENTOBI_GTP if DEFAULT_PENTOBI_GTP.exists() else None


class GtpError(RuntimeError):
    """Raised when the engine returns a ``?`` (failure) response or dies."""


class PentobiGtp:
    """A single ``pentobi-gtp`` engine process for Blokus Duo.

    Single-threaded per instance (Pentobi degrades past ~8 threads, and we run one
    process per concurrent game). Use as a context manager or call :meth:`close`.
    """

    def __init__(
        self,
        level: int,
        *,
        binary: str | Path | None = None,
        threads: int = 1,
        seed: int | None = None,
        game: str = "duo",
    ) -> None:
        path = Path(binary).expanduser() if binary else find_pentobi_gtp()
        if path is None or not path.exists():
            raise FileNotFoundError(
                "pentobi-gtp binary not found. Build it (docs/plans/pentobi-harness.md H2) "
                "or set $PENTOBI_GTP_PATH.",
            )
        argv = [str(path), "--game", game, "--level", str(level),
                "--quiet", "--threads", str(threads)]
        if seed is not None:
            argv += ["--seed", str(seed)]
        # stderr → DEVNULL: --quiet already silences logging; discarding stderr removes
        # any chance of a full-pipe deadlock. GTP errors arrive on stdout as "? ...".
        self._proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self.level = level
        # Sanity-check the engine is alive and is what we expect.
        if self.send("name") != "Pentobi":
            raise GtpError("unexpected engine — 'name' did not return 'Pentobi'")

    # -- transport -------------------------------------------------------------

    def send(self, command: str) -> str:
        """Send one GTP command and return its response payload (raises on ``?``)."""
        if self._proc.poll() is not None:
            raise GtpError("pentobi-gtp process is not running")
        assert self._proc.stdin is not None
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()
        return self._read_response()

    def _read_response(self) -> str:
        """Read one GTP response: skip leading blanks, collect until a blank line."""
        assert self._proc.stdout is not None
        lines: list[str] = []
        while True:
            raw = self._proc.stdout.readline()
            if raw == "":  # EOF — engine exited
                raise GtpError("pentobi-gtp closed its output (process died)")
            line = raw.rstrip("\n")
            if line == "":
                if lines:
                    break  # blank line terminates the response
                continue   # skip leading blank lines
            lines.append(line)
        status = lines[0]
        # Strip the leading '='/'?' from the first line; keep any continuation lines.
        body = "\n".join([status[1:].lstrip(), *lines[1:]]).strip()
        if status.startswith("?"):
            raise GtpError(body)
        return body

    # -- GTP commands ----------------------------------------------------------

    def clear_board(self) -> None:
        self.send("clear_board")

    def set_random_seed(self, seed: int) -> None:
        """Reseed the engine's RNG (GTP ``set_random_seed``).

        ``clear_board`` resets the position but *not* the RNG, so a reused engine
        replays one continuous random stream across games. Reseeding per game makes
        each game an independent draw (evaluation independence)."""
        self.send(f"set_random_seed {seed}")

    def play(self, color: str, move: str) -> None:
        """Inform the engine of a move. ``color`` is 'b'/'w'; ``move`` is cells or 'pass'."""
        self.send(f"play {color} {move}")

    def genmove(self, color: str) -> str:
        """Ask the engine to generate and play a move; returns cells or 'pass'."""
        return self.send(f"genmove {color}")

    def reg_genmove(self, color: str) -> str:
        """Generate a move without playing it (no state change)."""
        return self.send(f"reg_genmove {color}")

    def showboard(self) -> str:
        return self.send("showboard")

    def final_score(self) -> str:
        """Return the GTP score string, e.g. ``B+5`` (B = our White, W = our Black)."""
        return self.send("final_score")

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        if self._proc.poll() is not None:
            return
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write("quit\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self._proc.kill()
            self._proc.wait(timeout=5)

    def __enter__(self) -> PentobiGtp:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

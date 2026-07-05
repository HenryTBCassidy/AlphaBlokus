# Pentobi `resign` / non-coordinate GTP move handling

**What this covers.** The Pentobi benchmark crashes when the Pentobi engine resigns. GTP `genmove`
can return a normal move, `pass`, or **`resign`**; our translation layer handles the first two but not
`resign`, so a resign is parsed as a coordinate and blows up. Because **a resign is Pentobi conceding
the game — i.e. a win for our net** — this bug doesn't just crash: it *discards wins*, and it will
block the full nine-level ladder benchmark on the final net.

**Observed failure (2026-07-05).** Benchmarking a gen-44 net (large 192×12) vs Pentobi, levels 1–3
completed (net won 60%/65%/60%), then **level 4 crashed**:
```
File ".../pentobi/translation.py", line 70, in pentobi_to_coord
    return ord(token[0]) - ord("a"), int(token[1:]) - 1
ValueError: invalid literal for int() with base 10: 'esign'
```
`token = "resign"` → `token[0]='r'`, `int("esign")` → crash. The crash aborted the whole benchmark,
losing levels 4–9. (Pentobi resigning at level 4 is itself a signal the net was winning decisively.)

**Prerequisites:** none. This is a self-contained harness fix.

**Ground truth for the current code** (verified file:line anchors the executor will edit):
- `src/alphablokus/games/blokusduo/pentobi/gtp.py:131-133` — `genmove(color)` returns the raw GTP move
  string: cells, `"pass"`, or `"resign"`. (`pass`/`resign` are standard GTP `genmove` responses.)
- `src/alphablokus/games/blokusduo/pentobi/translation.py:67-70` — `pentobi_to_coord(token)` does
  `int(token[1:])` — the exact crash site for any non-coordinate token.
- `src/alphablokus/games/blokusduo/pentobi/translation.py:110-114` — `pentobi_to_action_index(move)`:
  handles `"pass"` (→ `pass_action_index`), else `cells_to_action` → crash on `"resign"`. `PASS = "pass"`
  constant at translation.py:31.
- `src/alphablokus/games/blokusduo/pentobi/player.py:98-105` — `PentobiPlayer.__call__` →
  `return self._translator.pentobi_to_action_index(self._engine.genmove(self._my_color))`. **This is the
  only place `genmove` output is consumed, so this is where a resign must be intercepted.**
- `src/alphablokus/games/blokusduo/pentobi/player.py:74-91` — `notify()` relays the *opponent's* moves;
  passes are skipped (not relayed). Resign never arrives here (it only comes from our own `genmove`), so
  `notify` needs no change — but confirm.
- `src/alphablokus/evaluation/arena.py:83` — `Arena.play_game(...) -> (outcome, GameRecord|None)`,
  `outcome` is `+1`/`-1`/small-float-draw **from player1's perspective**. The loop calls the current
  player, **asserts the returned action is legal**, applies it, checks game-over, then scores. Resign
  must be handled *before* the legality assert.
- `src/alphablokus/evaluation/arena.py:188` — `Arena.play_games(num, ...) -> (p1_wins, p2_wins, draws, records)`; swaps player1/player2 at halftime. As long as `play_game` returns the correctly-signed
  outcome, the tallies here stay correct.
- `scripts/pentobi_benchmark.py` — runs games via the Arena and tallies net W/L/D per level; it must
  count a resign as a **net win**.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| R1 | Reproduce + lock behaviour: unit test for `resign`/`pass`/move in translation (currently crashes) | 30 min | High | ✅ |
| R2 | Handle `resign` in the translation/player boundary + `RESIGN` constant; define the resign signal | 1 h | High | ✅ |
| R3 | Handle the resign signal in `Arena.play_game`: end game, award win to the opponent | 1.5 h | High | ✅ |
| R4 | Confirm `pentobi_benchmark.py` scores a resign as a net win; robust to no crash | 45 min | High | ✅ |
| R5 | Tests: Arena-level resign (opponent wins), translation resign, colour-swap correctness | 1.5 h | High | ✅ |
| R6 | End-to-end validate (`pentobi_benchmark --levels 4-9`, no crash, resigns = wins) + docs | 45 min | Medium | ✅ |

---

## R1. Reproduce and lock behaviour

**Current state.** `translation.py` has no test that feeds it `"resign"`; the crash is unguarded.

**Fix.** Add/extend `tests/games/blokusduo/pentobi/test_translation.py` (real `BlokusDuoGame`, no mocks):
- `pentobi_to_action_index("pass")` and `"PASS"`/`" pass "` → `pass_action_index` (lock existing behaviour).
- A normal move round-trips: `action_index_to_pentobi(a)` then `pentobi_to_action_index(...)` → `a` for a
  few sample actions (lock the happy path).
- `pentobi_to_action_index("resign")` — assert the **new** contract from R2 (raises a typed
  `PentobiResigned` / returns the resign sentinel), **not** a `ValueError` from `int()`. Write this test
  first to capture the bug (it currently raises the wrong error), then make it pass in R2.

**Effort:** 30 min.

---

## R2. Handle `resign` at the translation/player boundary

**Design decision (pick one; recommend the sentinel).** A resign is not a board move, so it cannot be a
normal action index. Two clean options:

- **(Recommended) A resign sentinel action.** Add `RESIGN_ACTION: int = -1` (a module constant in
  `arena.py` or `interfaces.py`) that is never a valid action index. `PentobiPlayer.__call__` returns it
  on resign; the Arena checks for it before the legality assert (R3). Simple, no exception control flow
  across the game loop.
- **(Alternative) A typed exception** `PentobiResigned(Exception)` raised by the player and caught in
  `Arena.play_game`. Cleaner semantics, but threads exception handling through the game loop.

**Fix (recommended path).**
- In `translation.py`: add `RESIGN = "resign"` alongside `PASS`. Make `pentobi_to_action_index` robust:
  if `move.strip().lower() == RESIGN`, do **not** try to parse coordinates. Since the translator returns
  an `int`, either (a) return `RESIGN_ACTION`, or (b) raise a clear typed error and let the *player*
  handle it. Prefer handling resign in the **player** (it owns the game-flow meaning), and in the
  translator just guard `pentobi_to_coord`/`cells_to_action` to raise a clear
  `ValueError(f"Non-coordinate GTP token {token!r}; caller must handle pass/resign")` instead of the
  cryptic `int()` error — defensive for any other caller.
- In `player.py` `__call__` (player.py:98-105): capture the raw genmove, branch on resign:
  ```python
  move = self._engine.genmove(self._my_color)
  if move.strip().lower() == RESIGN:
      return RESIGN_ACTION            # Arena scores this as a loss for us (Pentobi)
  return self._translator.pentobi_to_action_index(move)
  ```
  (Import `RESIGN` from translation, `RESIGN_ACTION` from wherever it's defined.)
- `notify()` needs no change (resign never arrives via the opponent path) — add a one-line comment
  confirming this.

**Effort:** 1 h.

---

## R3. Handle the resign signal in `Arena.play_game`

**Current state.** `play_game` (arena.py:83) calls the current player, asserts the returned action is
legal, applies it, loops until game-over, then scores by Blokus rules. A `RESIGN_ACTION` (-1) would
fail the legality assert.

**Fix.** In `play_game`, immediately after getting the action from the current player and **before** the
legality assert / applying the move:
```python
if action == RESIGN_ACTION:
    # current player resigns -> the OTHER player wins
    winner_is_player1 = (current_player is not player1)
    outcome = 1.0 if winner_is_player1 else -1.0
    return outcome, record   # record = moves so far (or None); optional
```
- Get the sign right relative to the `outcome`-from-player1's-perspective convention (arena.py) and
  whatever variable tracks whose turn it is. **Add a test (R5) that pins the sign.**
- The `GameRecord` for a resigned game can be the moves played so far (truncated) or `None` — the replay
  viewer should tolerate it; keep it minimal.
- `play_games` (arena.py:188) needs no change: it aggregates signed `play_game` outcomes and already
  swaps colours, so a correctly-signed resign outcome tallies correctly for both halves.

**Effort:** 1.5 h.

---

## R4. Confirm `pentobi_benchmark.py` scores a resign as a net win

**Fix.** Trace how `scripts/pentobi_benchmark.py` derives per-level net W/L/D from the Arena result
(it plays net vs `PentobiPlayer` via `Arena.play_games` and counts outcomes). Confirm that a resign —
now a normal signed win outcome from R3 — increments the **net win** count, not a draw/error. If the
benchmark anywhere calls `PentobiPlayer.final_score()` to decide the winner, ensure the resign path
(where there is no natural final score) is handled — prefer using the Arena outcome as the source of
truth for who won, and only use `final_score` for margin/reporting. Add a short log line when a game
ends by resign so it's visible in the benchmark output.

**Effort:** 45 min.

---

## R5. Tests

Real objects, no mocks (style guide). Key tests:
- **Arena resign, both sides (the important one).** Use two trivial callable players on a real
  `TicTacToeGame` (or `BlokusDuoGame`): one that returns `RESIGN_ACTION` on its first move, one that
  plays legal moves. Run `Arena.play_game` with the resigner as player1 → assert `outcome == -1`; then
  as player2 → assert `outcome == +1`. This pins the sign convention independent of Pentobi.
- **Translation resign** (from R1): `pentobi_to_action_index("resign")` behaves per the R2 contract;
  `pentobi_to_coord("resign")` raises the clear guarded error, not the `int()` `ValueError`.
- **Colour-swap correctness:** a small `play_games(n)` where one player always resigns → the resigner
  loses every game regardless of the halftime swap (all wins to the opponent).
- (Optional) If the `PentobiPlayer` can be constructed with an injectable engine stub that returns
  `"resign"`, add a player-level test; otherwise the Arena-level test above is sufficient and doesn't
  require a live Pentobi process. Injecting a fake GTP engine (an object with `genmove`/`play`/
  `clear_board`) is I/O stubbing, not game-logic mocking, so it's within the style guide.

**Effort:** 1.5 h.

---

## R6. End-to-end validation + docs

- Run `uv run python -m scripts.pentobi_benchmark --config run_configurations/blokus_cloud.json --net
  <a-net> --levels 4-9 --games 20` against a strong net and confirm: **no crash**, games that end by
  Pentobi resignation are counted as net wins, and a full ladder result is produced. (The gen-44 /
  final net from the `blokus_cloud_60` run is the natural test subject — it made Pentobi resign at
  level 4.)
- Update `docs/06-INTERFACES.md` (Pentobi GTP section) to document that `genmove` may return
  `pass`/`resign` and how each is handled (pass → pass action; resign → immediate loss for the
  resigner).
- Full CI green (ruff + format + mypy + tests).

**Effort:** 45 min.

---

## As-built notes / scope additions

- **Design chosen: sentinel via the translator** (R2 recommended option (a)). `RESIGN_ACTION = -1`
  lives in `alphablokus/interfaces.py`. `PentobiMoveTranslator.pentobi_to_action_index` returns it for
  `resign` (alongside `pass` → pass action). Because `PentobiPlayer.__call__` already returns whatever
  the translator produces, the player needed **no branch** — just a docstring clarifying the three
  `genmove` outcomes. `pentobi_to_coord`/`cells_to_action` now raise a clear
  `ValueError("Non-coordinate GTP token …")` defensively for any other non-cell token.
- **Arena** (`play_game`): intercepts `RESIGN_ACTION` immediately after the player returns, before the
  legality assert; outcome is `float(-cur_player)` (opponent wins, player1-perspective), runs `endGame`
  hooks, and `logger.info`s the resignation (visible in benchmark output — R4). `play_games` needed no
  change; the signed outcome tallies correctly through the halftime colour swap.
- **R4 was confirm-only** — no benchmark code change. The script already derives net W/L/D from the
  signed `Arena.play_games` tally (never from `final_score` to decide the winner), so a resign now
  increments `net_wins`. Traced + covered by the colour-swap test.
- **Extra test beyond the (optional) player-level item:** `test_pentobi_resignation_scores_as_net_win`
  in `test_player.py` drives the **real** `pentobi-gtp` binary + real `BlokusDuoGame` + real Arena,
  stubbing only the `genmove` I/O to return `resign` (both colours). This is the true end-to-end
  reproduction of the crash, gated on the binary (runs on the box, skips where absent).
- **R6 validation caveat:** the literal `--levels 4-9` run against the strong gen-44 `blokus_cloud_60`
  net needs the box (that net + its large 192×12 arch aren't local, and only a decisively-winning net
  makes Pentobi resign). Validated locally instead by: (a) the real-binary player resign test above,
  (b) a fresh-net `pentobi_benchmark --level 1` end-to-end run (no crash, ladder JSON + report produced),
  and (c) full unit coverage of the sign convention. The strong-net ladder run remains a cheap manual
  confirmation for the next box session.

## Notes for the executing agent

- **Style contract:** full type annotations (mypy `--strict`), `ruff` lint + format, frozen dataclasses,
  loguru (`{}` placeholders, no `print`), Google docstrings, `from __future__ import annotations`, real
  objects in tests (stubbing the external GTP subprocess is fine; don't mock game logic). Keep CI green.
- **Scope discipline:** this is a *harness correctness* fix — don't refactor the translator's shape-
  matching or the Arena loop beyond what's needed to handle resign (and, defensively, any other
  non-coordinate `genmove` token). `pass` already works; don't regress it.
- **One commit per checklist row**; tick the Done column the moment each row lands.
- **Why it matters:** without this, the definitive full-ladder benchmark on the final `blokus_cloud_60`
  net will crash the moment Pentobi resigns at any level — and silently under-counts the net's strength
  (resigns are wins). It's a prerequisite for a trustworthy Pentobi result.
- **Archive on completion:** when every row is ✅, `git mv` this file to `docs/plans/archive/` (per
  PLAN-FORMAT.md).

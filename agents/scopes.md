# Live scopes

Which Worker sessions exist, what each owns, and which files are theirs. The Integrator keeps this
current. Exclusive ownership is what stops two sessions fighting over one PR.

`src/alphablokus/config.py` is the one **shared** file — flag it in the PR title when you touch it.

| Scope | Plan + items | Owns these paths | Status |
|---|---|---|---|
| *(none yet — to be filled when the work is reorganised)* | | | |

## Reserved ownership, whoever takes these scopes

| Path | Scope that owns it |
|---|---|
| `scripts/pentobi_benchmark.py`, `scripts/measure_move_times.py`, `scripts/mini_ladder.py` | benchmark |
| `src/alphablokus/games/blokusduo/pentobi/**` | benchmark |
| `src/alphablokus/evaluation/ladder_*.py` | benchmark |
| `docs/plans/fair-pentobi-benchmark.md`, `docs/10-EVALUATION-SPEC.md` | benchmark |
| `src/alphablokus/selfplay/**`, `src/alphablokus/storage/selfplay_store.py` | value head |
| `src/alphablokus/training/**`, `src/alphablokus/games/base_wrapper.py` | value head |
| `src/alphablokus/cli.py` | value head |
| `docs/plans/supervised-network-improvements.md` | value head / architecture — **split before starting** |
| `src/alphablokus/games/blokusduo/nn/**` | architecture |
| `src/alphablokus/games/blokusduo/pentobi/corpus*.py`, `store.py` | corpus |
| `docs/plans/future/pentobi-corpus-v2.md` | corpus |

**Note:** `supervised-network-improvements.md` currently holds items belonging to at least two
scopes (value-head targets N6/N7, trunk architecture N8, and the score-head A/B N3 which duplicates
`score-auxiliary-target.md` S7). It must be split before two Workers touch it.

# Historical run configs

Verbatim copies of configs that produced results the project still cites, recovered from
the machines they ran on. They are **records, not runnable templates** — do not point a new
run at one.

They live here rather than in `run_configurations/` on purpose. `run_configurations/` is
validated by `tests/test_config_validation.py`, which rejects knobs the active search path
ignores (`dirichlet_epsilon` and `temp_threshold` under Gumbel). These files carry exactly
those knobs, because that is what actually ran. Fixing them would falsify the record;
exempting them from the validator would weaken it. So they are archived outside its scope.

| File | Run | Why it matters |
|---|---|---|
| `blokus_cloud_v3.json` | `blokus_cloud_v3`, 60 generations requested, reached 40 | The run that produced **gen-40**, the strongest net the project has measured (weighted 0.344 over Pentobi L1–9 at 400 sims, 100 games/level). Every experiment since warm-started from it. It was untracked on the box only — one disk failure from losing the provenance of the project's best result. |

Two further untracked box configs (`blokus_boxtest_paired.json`, `blokus_parity_check.json`)
were recovered at the same time but are not archived here: neither produced a cited result.
They sit in `temp/box_untracked_configs/` on the Mac if ever needed.

**Note on `num_generations`.** `blokus_cloud_v3.json` requests 60 generations; the run
stopped at 40. Do not read the config as a description of what completed — that is the
failure mode `run_provenance.json` and `config.resolved.json` now exist to prevent.

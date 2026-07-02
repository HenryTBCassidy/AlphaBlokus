"""GPU-native batched self-play (``selfplay_backend: "jax"``).

Structure (plan: ``docs/plans/jax-selfplay-pipeline.md``):

- :mod:`core.jaxplay.actors` — the jitted device loop: B games stepped in
  lockstep, each move searched by mctx (``games/blokusduo/jaxenv/search``),
  finished games auto-reset in place (pgx pattern), fixed-shape traces emitted.
- :mod:`core.jaxplay.harvest` — host-side: traces → completed games →
  ``ProcessedExample`` lists in the exact format the Coach/replay buffer/
  storage already consume (canonical compact boards, sparse policies, values,
  transpose augmentation).
- :mod:`core.jaxplay.backend` — the Coach-facing entry point
  ``generate_self_play_games`` (checkpoint load → weight conversion → waves
  until ``num_eps`` games harvested → games + stats).

Everything here imports jax lazily-by-placement: the modules are only imported
by ``Coach._run_self_play_jax`` behind the config flag, so python-backend runs
never require the ``jax`` extra.
"""

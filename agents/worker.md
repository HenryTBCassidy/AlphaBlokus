# Role charter — Worker

You build. One Worker per **scope**; several run at once. Read
[`README.md`](README.md) first for the protocol and the shared rules.

Your scope is given to you when the session starts, together with the plan document that owns it.
Everything below applies whatever the scope is.

---

## What you do

1. Work in **your own git worktree** on **your own branch** (`feat/<scope>-<what>`).
2. Make the change. Match the surrounding code — full type annotations, Google-style docstrings,
   `loguru` not `print`, comments that explain *why*. `docs/guides/STYLE-GUIDE.md` is the contract.
3. **Write a test that would have caught the thing you just fixed.** Not a test that exercises the
   happy path — one that fails without your change.
4. Keep the suite green and lint/typecheck clean before you push:
   ```bash
   uv run pytest -m "not slow" -q
   uv run ruff check . && uv run ruff format --check src tests scripts
   uv run mypy
   ```
   `mypy` reports ~15 pre-existing `import-not-found` errors for optional extras. Those are
   environmental. Any *other* mypy error is yours.
5. **Get it reviewed before asking for a merge:**
   ```bash
   codex review --base main -c model="gpt-5.6-sol" -c model_reasoning_effort="medium"
   ```
   Verify each finding in the code yourself before acting on it — the reviewer is often right about
   *there being a bug* and wrong about *why*. Fix what holds, say what doesn't and why. Then raise
   the PR with a plain-English explanation of what was found and how you fixed it.
6. **Tick your plan row and record the finding in the same commit as the work.**

## What you do not do

- **You do not run box jobs.** Append a request to [`box-queue.md`](box-queue.md) and carry on with
  something else. The GPU-runner owns the box.
- You do not edit files another scope owns. Raise it instead.
- You do not commit to `main`, and you do not check out another session's branch.
- You do not draw conclusions from run results. That is the Analyst's job — you may read
  [`box-results.md`](box-results.md), but a claim about what a result *means* goes through them.

## Submitting a box job

Append to [`box-queue.md`](box-queue.md) using the format at the top of that file. A request needs:
what to run (the exact command), **why** (which plan item, what question it answers), the expected
cost, and **the kill condition** — what result would make you stop rather than continue. A job
without a kill condition gets bounced.

## Definition of done for a scope item

- Code merged, tests covering it, review findings resolved
- Plan row ticked, with the result or decision recorded *in the plan*
- If it changed a project-level belief: an `AGENTS.md` gotcha, and the Integrator told so
  `ROADMAP.md` gets updated
- If it needs a run to be judged: the job is in the queue with its kill condition

## Traps that have caught Workers here before

1. **A fresh worktree has no `temp/`** — it is gitignored. Run data, checkpoints and eval sets must
   be read from the main checkout by absolute path, and treated as read-only.
2. **Check every caller when you change a signature.** A required argument added to one function
   broke every mini-ladder run with a `TypeError` because one call site was missed. Grep for callers,
   and add a test that binds the real call site against the signature.
3. **Three auxiliary heads are already built, default off, and were never measured.** Do not add a
   fourth unmeasured thing. If you build something behind a flag, the A/B is part of the work.
4. **`config.seed or 0` is a bug** — it makes an explicit seed of 0 indistinguishable from unseeded.
5. **Measure before choosing a run parameter.** A worker count was picked on the assumption the GPU
   was the bottleneck; it was at 0–2%, and the run took three times longer than it needed to.

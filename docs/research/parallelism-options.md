# Parallelism / Worker-Memory Options — Decision Doc

**The problem.** Self-play (MCTS + move-gen + tiny-net inference) is ~86% of the training cycle and CPU-bound. It's parallelised with a `ProcessPoolExecutor` (spawn) over 8 worker processes; each independently loads a full PyTorch+CUDA stack ≈ **2.5 GB RAM** (the net is only 340K params / 1.5 MB). So 8 workers ≈ 20 GB → the worker count is **capped at 8 even though the PC has 20 cores**, and a run just **OOM-crashed at the 28 GB cap** (at gen 4). We need to parallelise *without* the N× framework footprint — to use all cores and to scale beyond one box.

**The fundamental abstraction.** Every option is a point in three axes:
- **Concurrency primitive:** OS processes · OS threads (needs GIL-free execution) · accelerator-vectorisation · distributed actors
- **Hot-loop runtime:** CPython (GIL) · free-threaded CPython · compiled extension (Rust/C++) · JAX/XLA
- **Inference serving:** per-worker · central server · fused on-device

| | Strategy | Concurrency | Hot-loop runtime | Inference |
|---|---|---|---|---|
| **S1** | Cheaper processes + central inference | processes | CPython | central server |
| **S2** | Free-threaded Python (no-GIL) | threads | no-GIL CPython | shared in-process |
| **S3** | Compiled native-threaded core (Rust/C++) | threads | compiled ext | shared / embedded |
| **S4** | Accelerator-vectorised self-play (JAX+mctx+pgx) | GPU vectorisation | JAX/XLA | fused on-device |
| **S5** | Distributed orchestration (Ray) — *wraps* S1/S3 | distributed actors | any | central actor |

**Hardware targets** (the winner changes per target):
- **Mac (dev):** Apple M4 Pro, 14 cores, 24 GB unified, Metal/MPS (no CUDA), Python 3.13.
- **Home PC (prod):** 20 cores, RTX 3060 Ti (8 GB), 28 GB usable (WSL2), Python 3.12, CUDA.
- **AWS single-GPU:** e.g. g5 (A10G 24 GB) + 8–32 vCPU.
- **AWS multi-node GPU farm:** many GPUs across nodes.

---

## S1 — Cheaper processes + central inference (shrink the worker)

**What it is.** Keep multiprocessing, but stop each worker owning a GPU stack. Two flavours: **(a)** CPU-only workers (run the tiny net on CPU — no per-worker CUDA context; ~2.5 GB → a few hundred MB, or ~tens of MB with ONNX Runtime); **(b)** one central GPU process owns the net and serves *batched* inference to lightweight CPU-only workers (the parked **F5** shared-memory server — or a production server like Triton/TorchServe with dynamic batching).

**How in our codebase.** Largely **already built.** (a) The device is a single `net_config.cuda` flag (`config.py:54` → `base_wrapper.py`); set it false for the self-play pool and workers skip CUDA entirely — the win is essentially a *per-phase device split* (CPU self-play, GPU train/arena). (b) The F5 server already exists — `core/inference_server.py` + `core/inference_channel.py` (`InferenceClientNet` drops into `MCTS` unchanged); `parallel_self_play.py` already spawns it. The remaining work is to **decouple the server from the speed framing** (it was parked for giving no *speed* gain) and use it for the **memory/scaling** angle: workers torch-CUDA-free, one GPU net. On Linux/WSL, `fork()`+copy-on-write could also share the read-only move-gen tables.

**Complexity / effort.** Flavour (a) per-phase split: **~0.5–1 day.** (a)+ONNX Runtime workers: +2–3 days. Flavour (b) finish/validate F5 for memory: **~2–4 days.** Riskiest part: CPU inference (or the F5 synchronous round-trip) becoming the new bottleneck — pin ORT to 1 intra-op thread/worker, and measure (F5's no-speedup result was *because* inference is transfer-bound, not compute-bound).

**Benefits.** Directly removes the OOM + the 8-worker ceiling → feed all 20 cores (~2.5× self-play throughput headroom). It's the *prerequisite* for using all cores on every target, and the natural seam for later multi-node work. Not primarily a *speed* trick (the 66% pure-Python slice still needs more cores, which this unlocks).

**Pitfalls.** CPU inference can grow the 33% inference slice (mitigate: thread-pin, measure); `fork()`+CUDA is impossible (CUDA can't survive fork — only works for CPU-only workers); F5's synchronous per-leaf round-trip adds IPC latency (memory win holds even if speed is flat); CUDA MPS does *not* cut per-process memory (red herring here); ONNX export must match the net+masking exactly. Triton/MPS are premature on a single box.

**Per-hardware.** Mac: modest — workers are already CPU-only (no CUDA), so duplication is just torch-CPU; ONNX/CoreML could shrink it. **PC: highest value — this is the box that OOM'd; do this.** AWS single-GPU: good (saturate vCPUs against one GPU; Triton starts to fit). AWS farm: the central-inference seam becomes a network boundary (needs Triton/gRPC) — right architecture eventually, premature now.

**CV signal.** **Solid.** This is *actor–learner disaggregation* — the Seed RL / MiniZero pattern (cheap CPU self-play actors + a central batched-inference server). Reads as real ML-systems fluency (knowing inference is transfer-bound, why MPS doesn't help, fork-CoW limits).

---

## S2 — Free-threaded Python (no-GIL)

**What it is.** Free-threaded CPython (PEP 703; the `cp313t`/`cp314t` builds) removes the GIL, so MCTS runs as real parallel **threads in one process**, sharing one CUDA context + one net + the move-gen tables. The 8×2.5 GB duplication collapses to one ~2.5 GB stack + small per-thread trees.

**How in our codebase.** The **cleanest code delta of any structural option**, because `core/mcts.py` already holds all tree state per-instance (no shared mutable state) and `F2MoveGenerator` is stateless. Swap the `ProcessPoolExecutor` for a `ThreadPoolExecutor`, build one game+net and close over them (the whole `_worker_init` pickling dance disappears, and **the F5 IPC machinery becomes redundant** — threads just call the net directly). Two real changes: per-thread RNG (the global `np.random`/`torch.manual_seed` seeding would race — use per-thread `Generator`s), and an eager one-time build of the move-gen table singleton (it has a documented first-call race).

**Complexity / effort.** Small *code* (pool swap + RNG refactor, ~a day) — but **front-loaded by dependency risk, not code.**

**Pitfalls — the blocker.** **PyTorch dropped `cp313t` support** (manylinux removed 3.13t images May 2026; PyTorch's path is 3.14t). And **CUDA-enabled free-threaded torch wheels do not exist yet**: even on standard 3.14, PyTorch ships **CPU-only** (a CUDA-on-3.14 request was closed "not planned", Dec 2025). So on our CUDA boxes there is **no off-the-shelf free-threaded+CUDA torch** today — building from source against `cp314t`+CUDA in WSL2 is a real project. Also: ~5–10% single-thread no-GIL tax (3.14t); our own RNG thread-safety bug to fix.

**Per-hardware.** Mac: a free-threaded build is installable (already on 3.13) and FT torch CPU/MPS wheels exist — viable as a *prototyping* box, but MPS-limited and `cp313t` is dying. **PC: where it would pay off most (the OOM box) and where it's most blocked** — needs a 3.12→3.14t upgrade *and* a CUDA FT wheel that doesn't exist. AWS single-GPU: excellent fit *if/when* cp314t+CUDA wheels land. AWS farm: a per-node tool, composes under a distribution layer.

**CV signal.** **Very on-trend** — free-threading is *the* 2024–26 CPython story (PEP 703/779). "One GPU-resident net, many GIL-free threads driving parallel MCTS+inference in one process" is exactly what inference-serving / RL-infra teams are prototyping. The judgement to recognise it's currently wheel-blocked reads as signal too.

---

## S3 — Compiled native-threaded core (Rust or C++)

**What it is.** Move the CPU-bound hot loop into a compiled extension that **releases the GIL** and uses **native threads sharing one address space**. Attacks both the interpreter overhead on the ~66% Python slice (move-gen + UCB select + board-copy) *and* the per-process memory floor. Language (Rust+PyO3/maturin vs C+++nanobind/pybind11) is an implementation detail — Rust gives memory-safety + the stronger CV signal; C++ gives the mature ML/CUDA ecosystem. Two scopes: **(a)** whole MCTS+move-gen loop in the core (inference embedded via `tch-rs`/`ort`), **(b)** EnvPool-style batched *env* only, MCTS stays in Python over batches.

**How in our codebase.** `core/mcts.py` (the dict-based tree + UCB loop) and `games/blokusduo/movegen_runtime.py` (already table-driven — the easiest piece to port first as a contained proof) become the compiled core; Python keeps Coach/training/reporting. **Inference is the crux:** calling back into Python/torch re-acquires the GIL (re-introducing the contention you removed), so the *correct* design embeds libtorch (`tch-rs`) or ONNX Runtime (`ort`) in the core with one in-process batched-inference queue — i.e. essentially a **from-scratch AlphaZero engine.**

**Complexity / effort.** **High–very high** (multi-week). Move-gen port is tractable; MCTS port (virtual loss, bit-exact UCB, Dirichlet, deterministic descent) is hard; **building wheels across arm64-Mac / Linux+CUDA / AWS+CUDA** (maturin/scikit-build + libtorch/ort per platform) is as much work as the rewrite. Riskiest: **correctness equivalence** vs the validated Python (needs the golden-equivalence-test discipline), and keeping two model runtimes (train-PyTorch vs infer-libtorch/ORT) numerically in sync.

**Benefits.** Native threads share state (more cores, no per-process floor) *and* 10–50× on the tight interpreter-bound loops → Amdahl ceiling ~1.8–2.5× per game *on top of* core-count scaling. **Caveat:** F5 already removed the per-worker *net* copy, so S3's marginal *memory* win is smaller than the headline.

**Per-hardware.** Mac: good for threading, but no good native-inference path on MPS → dev/build box only. **PC: the only place it pays off — but B5 (free-threading) likely dominates it on effort for the same shared-memory threading, and a Cython/Numba pass gets the interpreter win cheaper.** AWS single-GPU: viable, rarely worth it over renting more cores + F5. AWS farm: out of scope (the bottleneck there is orchestration, not interpreter overhead).

**CV signal.** **Highest of any option** (esp. Rust+PyO3). It's the high-end pattern: AlphaGo/MuZero search in C++, KataGo C++/CUDA, Pentobi C++, EnvPool (Sea AI Lab), and the rising Rust+PyO3 wave (tokenizers, safetensors, Polars). The "compiled hot loop + Python research glue" split is *exactly* the quant production pattern. **But** the CV value is largely orthogonal to the project's goal (beat Pentobi) — flag that tension.

---

## S4 — Accelerator-vectorised self-play (JAX + mctx + pgx)

**What it is.** The frontier method: stop parallelising games across CPUs; run hundreds–thousands of games as one batched on-GPU computation. Env step + MCTS + inference all pure-functional JAX, `jit`+`vmap`'d across the batch. Libraries: **`mctx`** (DeepMind's batched MCTS in JAX — AlphaZero/Gumbel-MuZero; v0.0.6, Sept 2025), **`pgx`** (vectorised JAX board-game envs — **but no Blokus**, you'd write one against its base framework), net reimplemented in flax/optax.

**How in our codebase.** A **near-total rewrite of the self-play engine** — `mcts.py`, `self_play.py`, `parallel_self_play.py`, and the Blokus game/move-gen are replaced by JAX; only the *concepts/hyperparameters* and the static move tables survive. It's effectively a parallel second implementation, with the PyTorch path kept as the validated oracle.

**Complexity / effort.** **The biggest and riskiest by far** (weeks–months). The crux: **vectorising Blokus move-gen in fixed-shape JAX** — no variable-length candidate lists, no `set` dedup, no early exit. You rederive it as a dense masked computation over all ~13.7k moves every step (the static F2 tables are the right ingredient, so it's *feasible* but a from-scratch reimplementation), plus the pain of debugging jitted code and proving bit-equivalence.

**Decisive pitfall — it doesn't fit our GPU.** `mctx` stores **dense `[batch, nodes, num_actions]`** arrays (six of them). With Blokus's **17,837 actions**, batch=1024, ~129 nodes, fp32 ≈ **~57 GB** — vs the PC's **8 GB**. Even tiny batches blow past it. Blokus's huge action space is fundamentally hostile to mctx's dense tree; the 3060 Ti **cannot run this at any useful batch.** JAX-Metal (Mac) is experimental/unmaintained → CPU-only there.

**Per-hardware.** Mac: **flops** (no CUDA-JAX). PC: **flops hard** (action-space memory blowout on 8 GB). AWS single-GPU (24 GB): marginal — memory-throttled by the action space; hard to justify the rewrite vs just renting a box for the existing engine. **AWS multi-node farm: shines** — `pmap` across many large-VRAM GPUs is the regime it was built for. But that's far beyond "beat Pentobi at home", at the highest rewrite cost.

**CV signal.** **Strongest of all** — the literal frontier method, `mctx` is DeepMind's own code, JAX is the de-facto research stack. "Reimplemented a non-trivial game as a vectorised JAX env + ran batched mctx self-play across GPUs" is a top-tier CV line. Earned by the hardest, riskiest work, on hardware we'd have to rent, for a problem that fights the tooling.

---

## S5 — Distributed orchestration (Ray / cluster)

**What it is.** Not a within-box memory trick — an **orchestration layer** (Ray) that wraps the within-box strategies and scales them across machines: stateful **actors** (rollout workers), a zero-copy **object store** (weights shared per-node via `ray.put`), and an **autoscaling cluster** (one box → AWS multi-node). Ray was built for exactly this RL self-play producer/consumer loop.

**How in our codebase.** We **already built S5's central piece by hand** — the F5 `InferenceServer` was written transport-agnostic (`RequestSource`/`PendingRequest` protocols), so **Ray is a new transport, not a rewrite.** Migration: rollout work → `@ray.remote(num_cpus=1)` actors holding an `InferenceClientNet` whose source is a Ray handle; a `@ray.remote(num_gpus=1)` central inference actor wrapping `predict_batch`; trainer `ray.put`s weights per generation; the Coach loop swaps its executor. Use **Ray Core actors, not RLlib's AlphaZero** (deprecated contrib, doesn't fit our custom MCTS).

**Complexity / effort.** Moderate-high, mostly **ops not code** (cluster YAML, AMIs/CUDA, IAM, autoscaler, S3 checkpoints). Riskiest: distributed debugging + environment parity + preserving determinism off the ordered `pool.map`.

**Benefits.** The only option that crosses machine boundaries: multi-node fan-out, autoscaling (pay for the farm only during self-play), fault tolerance (a dead node reschedules instead of killing the run), zero-copy weight sharing.

**Pitfalls.** Overkill/net-negative on a single box; **per-actor memory is NOT solved by Ray alone** — it only works paired with the central inference actor (S1) so rollout actors are CPU-only; ops complexity + standing cluster cost; `ray.put` isn't auto-broadcast (fine for our 1.5 MB net).

**Per-hardware.** Mac / PC / AWS-single: **low value** (single node — the existing pool + F5 already covers it). **AWS farm: high value — the only place it earns its keep** (CPU rollout nodes on spot + 1–2 GPU inference nodes).

**CV signal.** **Strong** — Ray/RLlib is the industry-standard distributed-RL infra (Berkeley RISELab → PyTorch Foundation, 2025); quant funds use Ray/Dask for distributed Monte-Carlo + backtesting. "Built a Ray actor topology with a central GPU inference actor + autoscaling AWS cluster for distributed AlphaZero self-play" is a strong platform/infra line.

---

## Strategy × hardware matrix

Verdict = feasibility + expected payoff (✅ strong · 🟡 viable/marginal · ❌ poor/blocked).

| Strategy | Mac (M4 Pro, MPS) | Home PC (3060 Ti 8 GB, 20c) | AWS single-GPU (24 GB) | AWS multi-node farm |
|---|---|---|---|---|
| **S1** cheaper processes + central inference | 🟡 modest (already CPU-only) | ✅ **best fix now (mostly built)** | ✅ saturate vCPUs / Triton | 🟡 needs network transport |
| **S2** free-threaded Python | 🟡 prototype only (MPS, cp313t dying) | ❌ **blocked** (no CUDA FT wheel) | ❌→✅ blocked now; great when wheels land | 🟡 per-node, composes under S5 |
| **S3** compiled Rust/C++ core | 🟡 build/dev box (no MPS inference) | 🟡 works, but B5 likely cheaper | 🟡 viable, rarely worth it | ❌ orchestration is the real bottleneck |
| **S4** JAX + mctx + pgx | ❌ no CUDA-JAX | ❌ **8 GB can't hold 17,837-action trees** | 🟡 marginal (action-space throttled) | ✅ **shines (only here)** |
| **S5** Ray / cluster | ❌ overhead, no gain | ❌ overhead, no gain | ❌ single node | ✅ **the only multi-node answer** |

---

## Recommendation

The three goals (fast results · technical optimality · CV signal) and the hardware pull in different directions, so the honest answer is **sequenced**, not a single pick:

1. **Now — do S1 (CPU-only self-play workers + finish the F5 server).** It's ~80% already built, ~0.5–1 day for the quick win, **fixes the OOM and unlocks all 20 cores (~2.5× self-play headroom) on the actual prod box.** A strong net comes from *generations trained*, not architecture elegance — this gets you training again immediately. Decent CV signal (actor–learner disaggregation).

2. **Strategic structural fix — keep S2 (free-threading) as the target,** and track the PyTorch `cp314t`+CUDA wheel timeline. When it lands it's the cleanest long-term answer (tiny code delta, our MCTS is already thread-isolated) and a strong, on-trend CV line. Not actionable today (wheel-blocked).

3. **If CV / learning is explicitly the priority — S3 (Rust+PyO3 core)** is the standout portfolio artifact (compiled MCTS+move-gen with an embedded `tch`/`ort` inference queue). It's high-cost and largely orthogonal to "beat Pentobi", so treat it as a deliberate showpiece, not the fast path. Port the table-driven move-gen first as a contained proof.

4. **Only if we move to AWS — S5 (Ray) wrapping S1** for a multi-node farm; **S4 (JAX)** only on big/multi-GPU *and* accepting the move-gen-in-JAX rewrite + the action-space memory issue. Both are premature for the single-PC goal.

**The tension, stated plainly:** fastest-results (S1) and best-CV (S3/S4) diverge. The best of both worlds is **S1 now** (unblock + train), then **S3 as a deliberate CV side-project** *or* **S2 when unblocked** — rather than betting the immediate path on the expensive frontier rewrites, which our current hardware (8 GB GPU, no CUDA on the Mac) actively fights.

---

## Deep-dive: S1 vs S2 in plain terms, and the hybrid (S6)

### The setup (analogy)
Self-play = playing ~1000 games to generate training data. Each game needs two things: a **brain** (the neural net, lives on the GPU) to score positions, and a lot of **thinking** (the MCTS tree search + move generation — pure CPU grind). We have **20 CPU cores = 20 desks** for thinkers, and **one GPU = one brain**.

**Today (multiprocessing):** we hire 8 **workers**, and each one is a fully independent person who insists on carrying their *own complete copy of the brain plus all the heavy lab equipment* (the 2.5 GB PyTorch+CUDA stack) into their *own private office*. 8 people × a full equipment set fills the building (RAM) at 8 — even though there are 20 desks. They barely use the brain they're all lugging around. That's the waste, and it's what OOM-crashed us.

### S1 — central brain booth, thin workers
Put **one expert in a central booth** holding the single brain (one GPU process). The workers become **thin** — they just think (MCTS), and when they need a position scored they **pass a note through a window** to the booth, which answers many notes at once (batched). Each worker now carries almost nothing (~0.5 GB) → **20 thin workers fit easily → all 20 desks used.** They're still separate *people in separate offices* (processes, separate memory), they just stopped lugging the brain. **We already built the booth — that's F5.** No new technology required; works today on every machine.

### S2 — one open-plan office, shared brain
Instead of separate offices, put **everyone in one open-plan office sharing a single brain + one equipment set on a central table** (one process, many threads). Hugely memory-efficient — one set of everything for all 20 thinkers. The blocker is a Python rule called the **GIL = a single "talking stick": only the person holding it may think at any instant.** In a shared office that means no real parallel thinking — which is *exactly why we were forced into separate offices (processes) in the first place.* **Free-threaded Python removes the talking stick** → everyone in the open office thinks at once, sharing the one brain. The catch (today): **the GPU brain doesn't yet come in a version that works in the no-talking-stick office** — PyTorch has no CUDA + free-threaded build. So you can't put the *GPU* brain on the shared table yet.

### S6 — the hybrid you proposed (open-plan thinkers + a separate brain booth)
Your idea: *threads instead of processes, but farm the GPU calls out to a dedicated GPU worker.* Combine them — **open-plan office (free-threaded, shared) for the THINKERS, with the GPU brain kept in a SEPARATE booth (its own process).** The thinkers share their CPU equipment on the open table (memory win + all cores); they pass notes to the GPU booth for scoring (S1's central inference). **Why this dodges the blocker:** the GPU brain is *not* on the open-plan table — it lives in its own booth = a separate, *normal* Python process (regular torch+CUDA). So the open-plan office never needs the CUDA-+-free-threaded build that doesn't exist. **So yes — this is feasible on the PC today.** Coordinates: threads (free-threaded) · CPU runtime in-thread · central inference in a separate GPU process.
- **Cost/wrinkle:** the thinker office (free-threaded Python 3.14t) and the GPU booth (normal Python 3.12 + CUDA) are *different Python interpreters*, so the note-passing between them is cross-version IPC (named shared memory + a simple signal) — an adaptation of the existing F5 channel, not a rewrite, but real extra work.

### The key insight (this is the important bit)
**The memory/core win comes from "thin workers + a central GPU brain" — NOT from threads-vs-processes.** Threads (free-threading) are *marginally* leaner (one shared equipment set vs ~0.5 GB per process) and skip worker-to-worker note-passing — but **thin PROCESSES already get you to "20 cores used, one GPU, low memory" today, with zero blocker and no cross-version IPC** (S1 flavour b, all on Python 3.12, the F5 channel works as-is). So:
- **S1 (thin processes) is the whole win, shippable now.**
- **S6 (free-threaded threads) is NOT worth it until a CUDA + free-threaded PyTorch wheel ships.** The part of free-threading that would be a real win — the GPU net shared *inside* one process, no IPC — is exactly what's blocked. With the GPU stuck in a separate process either way, **S6 collapses to "S1 with threads instead of processes,"** and that swap trades unused memory headroom for a **~5–10% no-GIL single-thread tax** plus extra complexity (Python upgrade + cross-version IPC). Net: wash-to-slightly-negative for us.

Free-threading only becomes a genuine win when the wheel lets the GPU net live *inside* the free-threaded process (the blocked pure-S2): then all the IPC and per-worker duplication vanish and the small no-GIL tax is easily repaid. **Until then: just S1.** (We can still cheaply *spike* free-threaded Python + numpy on the CPU side any time, to be ready — but it's not on the critical path.)

### Correction: there *is* a near-universal option — **S1**
Re-reading the matrix: **S1 is the only strategy with no ❌** — it degrades gracefully on all four targets (best on the PC, fine on AWS-single, modest on Mac, the right seam for the farm). The "no single option works everywhere" impression was overstated; S1 is the universal baseline, and S6 is its upgraded form where free-threading is available.

### Is pure-S2 *really* impossible on the PC today?
Not "impossible" — **blocked off-the-shelf.** The only ways to put the *GPU* brain in a free-threaded process right now are: (1) build PyTorch from source against `cp314t`+CUDA (multi-day, fragile, in WSL2), or (2) use a non-PyTorch CUDA inference runtime in-thread (e.g. ONNX Runtime's CUDA EP, if it ships free-threaded wheels). Both are spikes, not sure things. **The hybrid (S6) avoids needing either** by keeping the GPU out of the free-threaded process. And we can validate free-threading itself *any time* with a CPU-only spike (free-threaded Python + numpy both exist) — that de-risks S6 cheaply.

## Sources
- **S1:** [Triton dynamic batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) · [CUDA MPS](https://docs.nvidia.com/deploy/mps/latest/index.html) · [PyTorch fork+CUDA #40403](https://github.com/pytorch/pytorch/issues/40403) · [MiniZero arXiv:2310.11305](https://arxiv.org/pdf/2310.11305)
- **S2:** [PyTorch dropping 3.13t](https://dev-discuss.pytorch.org/t/dropping-python-3-13t-free-threaded-support-in-nightlies-and-in-the-future-pytorch-2-13-release/3386) · [no CUDA 3.14 wheels #169929](https://github.com/pytorch/pytorch/issues/169929) · [py-free-threading tracking](https://py-free-threading.github.io/tracking/) · [3.14 free-threading benchmarks](https://www.danilchenko.dev/posts/python-314-free-threading/)
- **S3:** [PyO3 0.28 + free-threading](https://pyo3.rs/v0.28.3/free-threading) · [tch-rs](https://github.com/LaurentMazare/tch-rs) · [nanobind benchmarks](https://nanobind.readthedocs.io/en/latest/benchmark.html) · [EnvPool](https://github.com/sail-sg/envpool)
- **S4:** [mctx](https://github.com/google-deepmind/mctx) + [tree.py (dense arrays)](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/tree.py) · [pgx](https://github.com/sotetsuk/pgx) · [turbozero](https://github.com/lowrollr/turbozero) · [JAX-Metal](https://developer.apple.com/metal/jax/) · [AlphaZero-on-Tablut arXiv:2604.05476](https://arxiv.org/html/2604.05476v1)
- **S5:** [RLlib](https://docs.ray.io/en/latest/rllib/index.html) · [Ray object store / zero-copy](https://docs.ray.io/en/latest/ray-core/objects/serialization.html) · [Ray clusters on AWS](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/aws.html) · [muzero-general](https://deepwiki.com/werner-duvaud/muzero-general/2-core-system-architecture)

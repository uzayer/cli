# ADR-0014: Memory search sizes itself to the machine

**Status:** Accepted (2026-10-19, at the user's request: "some users have a
GPU, some have very low-end hardware, some have no GPU; make the CLI run
smoothly for all of them"). Amends the embedder of v0.4.0.

## Context

The MiniLM embedder runs in pure Go on every core. On a small server it was
the CLI's biggest cost: each process that embedded read the 87 MB weights
and copied them into float32 (177 MB allocated, 89 MB kept), a forward pass
took every core, and hooks started a detached index pass after every
session and an extraction after every turn, each loading its own copy. Two
constraints bound the fix:

- Release builds are `CGO_ENABLED=0` (ADR-0011), so the CLI cannot link
  CUDA or Metal itself.
- The vectors are the app's: `embed_cache` is keyed by model and text and
  both sides read what the other wrote. Another model on small machines
  would split the cache, so the cheap tier is keywords only (BM25, which
  already works without a model), not a smaller model.

## Decision

1. **The CPU path is cheap everywhere.** The weights are memory-mapped and
   used in place (every release target is little-endian), so loading
   allocates 4 MB and every atlas-cli process shares one copy in the page
   cache. The linear kernel takes rows in cached blocks, four per weight
   row: 1.5–1.7× faster on one core, the same vectors to 1e-7.
2. **Background passes yield.** A detached index pass or extraction runs at
   nice 10 (every thread on Linux), on half the cores (one on two or
   fewer), and one at a time across all Projects (a machine-wide lock in
   the CLI's state). A hook that asks for a pass while one runs only marks
   the Project; the running pass goes once more.
3. **A compute plan per machine**, in `<state>/compute.json`, set with
   `atlas-cli memory compute` or Settings ▸ Model, overridden by
   `ATLAS_EMBED`, `ATLAS_EMBED_MODEL`, `ATLAS_EMBED_THREADS`:
   - `auto` (default): this machine's CPU; keywords only on a small machine
     (one core, or under 2 GB, a container's limits included); a local
     Ollama serving `all-minilm` when there is one.
   - `off`, `cpu`: keywords only, or the CPU whatever the machine.
   - `ollama` or an OpenAI-compatible URL (TEI, vLLM, LM Studio): a GPU
     process does the forward pass. The endpoint is called without a key.
4. **An endpoint is used only once it embeds as the app does.** Four probe
   texts (up to past the 512-token cut, which catches an endpoint that
   truncates at 256) must each come back within cosine 0.995 of vectors
   stored in the package. The verdict is kept for a month, a mismatch for a
   day, an endpoint that accepted the connection but failed or hung for two
   minutes; `atlas-cli memory compute` always checks afresh. A recorded match
   is used only while the endpoint's port answers a dial. An endpoint that
   fails at runtime gives the rest of the process to the CPU, or to keywords
   on a small machine.
5. **A damaged model is said, and mended.** Model files that do not load
   make status and Settings say "keywords only" with the reason; a
   foreground `atlas-cli memory index` removes and downloads them again
   (a download alone keeps files that exist).

## Consequences

- A small server no longer downloads or runs the model unless told to;
  `atlas-cli status` says why search is keywords only.
- The desktop app is not used as an accelerator: it exposes no embedding
  API the CLI could call. If it gains one, it is another endpoint.
- A non-loopback endpoint receives memory text; `atlas-cli memory compute`
  says so when it is set.

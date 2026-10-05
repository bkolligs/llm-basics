# LLM basics — build it, break it, explain it

A **uv project with self-contained marimo exercises** for learning the LLM stack. The progression follows [Ahmad Osman’s Step-By-Step LLM Engineering Projects](https://x.com/TheAhmadOsman/article/2058745340895870985). These are original exercises inspired by the roadmap, not a reproduction of the article.

The first six labs are authored. Each supplies infrastructure and leaves the essential algorithms for you to implement. Later projects are a roadmap, not empty notebooks. Assume Python and basic tensor familiarity.

## Start locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from this folder:

```bash
uv sync --locked
uv run marimo edit
# Or open the first exercise directly:
uv run marimo edit notebooks/01_tokenization.py
```

Python 3.12 is the local default. The project supports 3.12–3.13 and locks dependencies for macOS and Linux. The first sync downloads dependencies; the exercises themselves use only bundled, originally authored synthetic data and need no API keys or model downloads.

Use **edit mode** to fill TODO functions and explanation cells. Infrastructure cells start collapsed; expand them when you want to inspect or adjust an experiment. If your marimo settings disable startup execution, click **Run all** once to initialize the controls and checks; experiment buttons still prevent training. A read-only app view is useful for inspecting results, but not completing exercises. Source files under `notebooks/` are the durable record of your implementations and Markdown answers.

## The Feynman loop

1. **Explain:** teach the idea to a curious beginner without relying on unexplained terminology. Give a concrete example and identify gaps.
2. **Predict:** write your expected result and why, before executing an experiment.
3. **Implement:** replace the marked `NotImplementedError` statements. Contracts specify shapes and conventions.
4. **Check and plot:** checks react to edits. Once they pass, press **Run fresh experiment** to produce three views of behavior.
5. **Break:** inspect the supplied ablation and failure gallery; change one variable, rerun, and explain the evidence.
6. **Explain again:** revise your account, work through an example, and identify the observation that changed your understanding.

Hints are progressively revealed, with no reference solutions. Tests cover selected behaviors rather than proving correctness. Completion means passing checks **and** interpreting plots, documenting a failure, and revising your explanation. Each lab is independent; later labs supply prerequisite components and may consequently demonstrate ideas from earlier labs.

Training and result generation are explicitly gated. Changing the device or exercise checks recreates the run control and invalidates dependent outputs; it does not launch training. Every experiment starts a fresh model and optimizer. Core checks run on the selected device but are deliberately tiny.

## Run the same notebook on Molab CUDA

[Molab](https://molab.marimo.io) runs server-backed marimo notebooks with an optional NVIDIA GPU. See its [official instructions](https://docs.marimo.io/guides/molab/).

1. Import/upload one of the notebook `.py` files into your Molab workspace. If this project is later hosted on GitHub, use Molab’s **new notebook → GitHub** import with the notebook URL. Fork a synced preview into an editable workspace to keep your answers.
2. Choose **server-backed execution**, not WebAssembly. Enable the GPU using the **notebook specs** control in the header.
3. Check the active kernel’s dependencies using Molab’s package manager. The local `pyproject.toml` is authoritative; Molab does **not** necessarily apply this repository’s `uv.lock`. Run `uv run python scripts/molab_requirements.py` locally to print the resolved top-level package versions for comparison. Use the package manager to align missing/incompatible packages; preserve a CUDA-enabled PyTorch build compatible with the attached GPU. No source-code changes or local support package are needed.
4. In the notebook select **auto** (CUDA → MPS → CPU) or **cuda** explicitly. The runtime panel reports the actual backend, GPU, PyTorch version, and dtype. Explicit CUDA selection gives actionable feedback if CUDA is unavailable. A detected CUDA runtime is not a guarantee that every installed kernel supports the attached GPU; run the small checks first.
5. Complete the exercise, then click **Run fresh experiment**. Download the notebook to preserve code and Markdown explanations. Export JSON measurements and the three-panel PNG using the supplied buttons.

The same files work locally with **mps** or **cpu**. Tensor allocation follows the chosen device; plots move results to CPU. Float32 and small workloads are the default everywhere. Python tokenization naturally stays on CPU. Accelerator timing includes synchronization; training timing also includes reporting overhead, so these are educational measurements, not optimized throughput benchmarks. Fixed seeds provide repeatable fixtures, not bitwise-identical results across hardware.

There is no GitHub remote configured yet, so no fabricated “Open in Molab” links are included. Once a repository URL is available, add each real notebook URL using Molab’s badge generator. Publishing is separate from local setup. Molab notebooks can be public but undiscoverable; use only these synthetic fixtures when sharing.

## Progression

**Ready** means exercise scaffolding, checks, plots, hints, and ablations are authored; it does not mean your TODOs are complete. Prerequisites are conceptual and never require importing another notebook’s answers. Numbers preserve Ahmad’s project order. Every row after 06 is **planned**.

| # | Project / notebook | Core learning objective | Conceptual prerequisites | Status |
|---|---|---|---|---|
| 01 | [Tokenization](notebooks/01_tokenization.py) | Byte-level BPE, reversibility, vocabulary/compression trade-offs | Python sequences, UTF-8 | Ready |
| 02 | [Embeddings](notebooks/02_embeddings.py) | One-hot equivalence, lookup gradients, learned geometry | 01, matrix multiplication | Ready |
| 03 | [Position](notebooks/03_position.py) | Sinusoids, learned tables, RoPE, ALiBi | 02, broadcasting | Ready |
| 04 | [Attention](notebooks/04_attention.py) | Scaling, stable softmax, causal masks, value aggregation | 02, dot products | Ready |
| 05 | [Multi-head attention](notebooks/05_multi_head.py) | Head layout, projection, composition and ablation | 04 | Ready |
| 06 | [Decoder block](notebooks/06_decoder_block.py) | Pre-norm residual paths, RMSNorm, SwiGLU | 03–05, gradients | Ready |
| 07 | Mini-former | Stack blocks and understand a full training loop | 06 | Planned |
| 08 | Learning objectives | Compare causal, masked, prefix and denoising objectives | 07 | Planned |
| 09 | Sampling dashboard | Temperature, truncation, entropy and generation failures | 07–08 | Planned |
| 10 | Speculative decoding | Draft/verify sampling and distribution preservation | 09 | Planned |
| 11 | KV cache | Incremental decoding, parity and memory use | 05, 09 | Planned |
| 12 | MQA, GQA and MLA | Shared KV heads and latent compression trade-offs | 11 | Planned |
| 13 | Sliding windows and attention sinks | Retention and bounded context | 11–12 | Planned |
| 14 | Context extension | RoPE interpolation, YaRN-style scaling and memory | 03, 13 | Planned |
| 15 | Efficient attention | Naive attention, SDPA and FlashAttention comparisons | 04, 11 | Planned |
| 16 | Hardware budget | FLOPs, bandwidth, memory and precision | 07, 11, 15 | Planned |
| 17 | Two-expert router | Sparse dispatch and balancing | 06–07 | Planned |
| 18 | Sparse-model trade-offs | Active versus total parameters, routing collapse | 16–17 | Planned |
| 19 | Alternative sequence models | Toy state-space or linear-attention recurrence | 04, 07 | Planned |
| 20 | Diffusion-style language modeling | Masking and iterative denoising | 08–09 | Planned |
| 21 | Pretraining data pipeline | Filtering, deduplication and split integrity | 01, 07 | Planned |
| 22 | Synthetic data | Generate, filter and evaluate incremental value | 21 | Planned |
| 23 | Scaling curves | Fit loss versus parameters, tokens and compute | 07, 16, 21 | Planned |
| 24 | Post-training | SFT, instruction tuning, LoRA and preference tuning | 07–08, 21 | Planned |
| 25 | RLHF / PPO / GRPO / RLVR | Rewards, advantages, policy updates and verification | 24 | Planned |
| 26 | Quantization | Quantize/dequantize and measure numerical/quality costs | 07, 16 | Planned |
| 27 | Serving systems | Scheduling, caching and comparable runtime measurements | 09–12, 16, 26 | Planned |
| 28 | Evaluation harness | Metrics, uncertainty, contamination and failure slices | 07, 21 | Planned |
| 29 | RAG | Retrieval, grounding and retrieval/generation error separation | 02, 09, 28 | Planned |
| 30 | Tools and agent loops | Tool interfaces, termination and trajectory evaluation | 28–29 | Planned |
| 31 | Tiny vision-language adapter | Connect visual representations to a language model | 02, 07, 24 | Planned |
| 32 | Interpretability | Probes, circuits and sparse autoencoders | 05–07, 28 | Planned |
| 33 | Safety evaluation | Controlled adversarial tests and failure documentation | 28, 30 | Planned |
| 34 | Integrated capstone | Train, tune, quantize, serve, evaluate and document | 07, 24, 26–30, 33 | Planned |

Later CUDA-specific kernels and serving stacks will be capability-gated extensions, while portable versions teach the underlying mechanism. Keep evaluation hygiene from the first notebook even though the dedicated harness appears later in the source roadmap.

## Validation and maintenance

```bash
uv run marimo check notebooks/*.py
uv run pytest
# Produce a clean-state HTML preview (unfinished TODOs are expected):
uv run marimo export html notebooks/01_tokenization.py -o artifacts/tokenization.html
```

Repository tests validate infrastructure, exercise contracts, and the clean unfinished state; they do not contain completed learner algorithms. During authoring, temporary reference implementations are used to verify experiments and then discarded from the delivered tree. Validation evidence and hardware limitations are recorded in [VALIDATION.md](VALIDATION.md).

Results downloaded from notebooks can be kept in `artifacts/`, which is ignored. Saving or downloading the `.py` file preserves your answers; JSON/PNG exports contain evidence, not your editable solution. Keep source and results together when reviewing a completed lab.

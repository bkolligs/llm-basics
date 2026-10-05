# LLM basics — build it, break it, explain it

Self-contained marimo exercises following [Ahmad Osman’s LLM engineering roadmap](https://x.com/TheAhmadOsman/article/2058745340895870985). Infrastructure is supplied; you implement the core algorithms. The first six labs are ready, with hints, checks, plots, and ablations. Python and basic tensor familiarity are assumed.

## Start locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync --locked
uv run marimo edit notebooks/01_tokenization.py
```

Use `uv run marimo edit` to browse all notebooks. Python 3.12 is the default; 3.13 is also supported. Exercises use bundled synthetic data and need no API keys or model downloads.

## How to learn

**Explain → predict → implement → check and plot → break → explain again.**

Write your predictions and explanations in the Markdown cells, fill the TODOs, then pass the checks and click **Run fresh experiment**. Inspect an ablation and revise your explanation using the evidence. Hints are included; solutions are not.

Labs are independent, with infrastructure cells collapsed. If startup execution is disabled, click **Run all** to initialize; training still requires the experiment button. Save the `.py` file to keep your code and answers, and use the export buttons for JSON results and PNG plots.

## Devices and Molab

Select **auto**, **cuda**, **mps**, or **cpu** in any notebook. Auto prefers CUDA → MPS → CPU; the runtime panel shows the selected backend. Changing devices resets the experiment control. All labs default to small float32 workloads.

To use [Molab](https://docs.marimo.io/guides/molab/):

1. Upload a notebook `.py` file or import its GitHub URL into an editable workspace.
2. Use server-backed execution and enable the GPU in **notebook specs**.
3. Check dependencies in Molab’s package manager; it may not apply `uv.lock`. Run `uv run python scripts/molab_requirements.py` locally to list the project’s locked versions. Use a CUDA-enabled PyTorch build compatible with the GPU.
4. Select **cuda**, run the checks, and start the experiment. Download the notebook to preserve your work.

CPU and MPS have been tested; Molab CUDA execution remains unverified.

## Progression

**Ready** means the exercise is authored, with TODOs for you to complete. Prerequisites are conceptual; labs do not import earlier answers.

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

## Project checks

```bash
uv run marimo check notebooks/*.py
uv run pytest
```

Keep exported results in `artifacts/` (ignored by version control).

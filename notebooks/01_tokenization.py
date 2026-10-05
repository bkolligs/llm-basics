# Self-contained learner notebook. Dependencies are managed by the root uv project.
# No inline dependency metadata: uv should use the project environment locally.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import json
    import time
    import io

    return io, json, mo, np, plt, time, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 01 · Text becomes tokens

    Build a byte-level BPE tokenizer. Start with all 256 byte tokens, so unfamiliar Unicode remains representable. Learn merges only on the tiny training corpus, then compare compression on held-out strings. Prerequisites: Python lists, dictionaries, UTF-8.

    **Workflow:** explain → predict → implement → check and plot → break → explain again.

    This lab is independent. Infrastructure and prerequisite components are supplied; only the marked exercise functions are yours. No completed solutions are included.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1 · Explain before coding

    **Edit this Markdown cell in the editor and save the notebook.** These answers live in the `.py` source, not transient widgets.

    - My explanation to a curious beginner: **[write here]**
    - A small concrete example: **[write here]**
    - What I cannot yet explain without jargon: **[write here]**

    ## 2 · Predict

    Will more merges always reduce held-out token counts? Which sample will have the most tokens per visible character? Why can a larger vocabulary increase memory use elsewhere?

    My predictions and reasons: **[write here before running]**
    """)
    return


@app.cell(hide_code=True)
def _(np, time, torch):
    def resolve_device(requested, backend=torch):
        """Infrastructure: auto prefers CUDA, then MPS, then CPU. Never silently downgrade explicit requests."""
        available = {"cpu": True, "cuda": backend.cuda.is_available(),
                     "mps": backend.backends.mps.is_available()}
        if requested == "auto":
            requested = next(name for name in ("cuda", "mps", "cpu") if available[name])
        if requested not in available:
            raise ValueError(f"Unknown device: {requested}")
        if not available[requested]:
            raise RuntimeError(f"{requested.upper()} is unavailable. In Molab, enable GPU in notebook specs and restart the kernel if needed. Check the active PyTorch installation, or choose CPU explicitly.")
        return backend.device(requested)

    def synchronize(device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    def timed(device, operation):
        synchronize(device)
        started = time.perf_counter()
        result = operation()
        synchronize(device)
        return result, time.perf_counter() - started

    def to_numpy(value):
        return value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

    def seeded_tensor(shape, device, seed=7):
        # CPU-generated fixtures make initial inputs identical across backends.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)

    def check_suite(cases):
        rows = []
        for name, check in cases:
            try:
                check()
            except NotImplementedError:
                rows.append({"check": name, "status": "TODO", "detail": "Implement the named exercise function, then rerun its cell."})
            except Exception as error:
                rows.append({"check": name, "status": "FAIL", "detail": f"{type(error).__name__}: {error}"})
            else:
                rows.append({"check": name, "status": "PASS", "detail": "Behavior verified on this fixture."})
        return rows

    def require(condition, message):
        if not bool(condition):
            raise AssertionError(message)

    def close(actual, expected, message="Values differ", atol=2e-5, rtol=2e-4):
        require(actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}")
        require(torch.allclose(actual, expected, atol=atol, rtol=rtol), message)

    return check_suite, require, resolve_device


@app.cell(hide_code=True)
def _(mo):
    device_choice = mo.ui.dropdown(options=["auto", "cuda", "mps", "cpu"], value="auto", label="Execution device")
    device_choice
    return (device_choice,)


@app.cell(hide_code=True)
def _(device_choice, mo, resolve_device, torch):
    try:
        device = resolve_device(device_choice.value)
        runtime_error = None
    except (RuntimeError, ValueError) as error:
        device = None
        runtime_error = str(error)
    mo.stop(runtime_error is not None, mo.callout(runtime_error or "", kind="warn"))
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else ("Apple Metal GPU" if device.type == "mps" else "CPU")
    mo.md(f"**Runtime:** `{device}` · {gpu_name} · PyTorch `{torch.__version__}` · `float32`\n\nChanging device clears the run control; start a new experiment explicitly. Pure Python tokenization stays on CPU.")
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · Implement

    Implement the contract below. `merges` is an ordered list of `(left_id, right_id, new_id)` triples; new IDs start at 256. Pair ties break lexicographically. Merge non-overlapping occurrences left to right. Training stops when no adjacent pairs remain. Encoding applies learned merges in training order. Decoding expands merge tokens recursively into bytes and decodes UTF-8 strictly. Only round trips of valid UTF-8 inputs are required.

    Replace each `raise NotImplementedError` in the next cell. Keep the function signatures. Allocate tensors on the input tensor’s device; the supplied fixtures already use the selected backend.
    """)
    return


@app.cell
def _():
    def pair_counts(ids):
        """list[int] -> dict[(int, int), int]; count overlapping adjacent pairs."""
        raise NotImplementedError("pair_counts")

    def merge_pair(ids, pair, new_id):
        """Replace non-overlapping occurrences; return a new list without mutating ids."""
        raise NotImplementedError("merge_pair")

    def train_bpe(text, num_merges):
        """str, nonnegative int -> ordered merge triples. Start from text.encode('utf-8')."""
        raise NotImplementedError("train_bpe")

    def encode(text, merges):
        """str, ordered merge triples -> list[int]. Empty text gives an empty list."""
        raise NotImplementedError("encode")

    def decode(ids, merges):
        """list[int], ordered merge triples -> str; invert the byte-level encoding."""
        raise NotImplementedError("decode")

    return decode, encode, merge_pair, pair_counts, train_bpe


@app.cell(hide_code=True)
def _(decode, encode, merge_pair, pair_counts, require, train_bpe):
    def check_pairs():
        require(pair_counts([1, 1, 1, 2]) == {(1, 1): 2, (1, 2): 1}, "Count overlapping pairs.")
        require(pair_counts([]) == {}, "Empty input has no pairs.")

    def check_merge():
        source = [1, 1, 1, 1, 1]
        require(merge_pair(source, (1, 1), 256) == [256, 256, 1], "Merges must not overlap.")
        require(source == [1, 1, 1, 1, 1], "Do not mutate input.")

    def check_training():
        require(train_bpe("abab", 1) == [(97, 98, 256)], "Most frequent pair should win.")
        require(train_bpe("baba", 2) == [(98, 97, 256), (256, 256, 257)], "Update counts after every merge.")
        require(train_bpe("aba", 1) == [(97, 98, 256)], "Break equal-count ties lexicographically.")
        require(train_bpe("", 4) == [], "Stop when no pair remains.")

    def check_encoding():
        rules = [(97, 98, 256), (256, 256, 257)]
        require(encode("abab!", rules) == [257, 33], "Apply rules in learned order.")
        require(encode("", rules) == [], "Empty input.")

    def check_roundtrip():
        rules = train_bpe("banana bandana banana!", 12)
        for sample in ["", "banana", "new words", "你好 🦙 café", "x = 12\n"]:
            require(decode(encode(sample, rules), rules) == sample, f"Round trip failed: {sample!r}")

    exercise_cases = [("Adjacent pair counts", check_pairs), ("Non-overlapping merge", check_merge), ("Greedy training and ties", check_training), ("Ordered encoding", check_encoding), ("UTF-8 round trips", check_roundtrip)]
    return (exercise_cases,)


@app.cell(hide_code=True)
def _(check_suite, exercise_cases, mo):
    check_results = check_suite(exercise_cases)
    checks_passed = all(row["status"] == "PASS" for row in check_results)
    mo.vstack([mo.md("## 4 · Behavioral checks"), mo.ui.table(check_results, selection=None)])
    return check_results, checks_passed


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Hint 1': 'Count neighboring IDs, including overlapping pairs; choose a deterministic winner.', 'Hint 2': 'After a merge, scan the updated sequence again. A consumed pair advances the scan by two.', 'Hint 3': 'Treat every byte as a base vocabulary item. Keep merge order distinct from pair frequency.'})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · Run, plot, and break

    **Ablation:** compare the learned merge list with an empty list. Inspect code, Unicode, and rare words. Why can valid round trips coexist with poor compression? Change the corpus in `experiment()` and rerun to test domain mismatch.

    The three panels are distinct measurements. Record failures as evidence, including a case where your prediction was wrong. Tiny synthetic tasks demonstrate mechanisms; they do not establish real language-model quality.
    """)
    return


@app.cell(hide_code=True)
def _(checks_passed, device, mo):
    # Depend on device and checks so changing either recreates this one-shot button.
    mo.stop(not checks_passed, mo.callout("Finish the TODOs and pass the checks to unlock experiments.", kind="info"))
    run_experiment = mo.ui.run_button(label=f"Run fresh experiment on {device}")
    run_experiment
    return (run_experiment,)


@app.cell(hide_code=True)
def _(decode, encode, np, plt, train_bpe):
    def experiment():
        corpus = "banana bandana banana. the cat sat on the mat. the dog sat on the mat. " * 12
        held = {"familiar": "the cat sat on the mat.", "code": "def f(x): return x**2", "unicode": "你好 🦙 café", "rare": "quizzical zephyrs"}
        budgets = [0, 4, 8, 16, 32, 64]
        rulesets = [train_bpe(corpus, n) for n in budgets]
        counts = [len(encode(corpus, rules)) for rules in rulesets]
        vocab = [256 + len(rules) for rules in rulesets]
        learned = rulesets[-1]
        ratios = [len(s.encode("utf-8")) / max(1, len(encode(s, learned))) for s in held.values()]
        tokens = encode(corpus, rulesets[3])  # 16 merges: before the entire corpus becomes one token.
        _, frequencies = np.unique(tokens, return_counts=True)
        _, byte_frequencies = np.unique(encode(corpus, []), return_counts=True)
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        axes[0].plot(vocab, counts, marker="o"); axes[0].set(xlabel="Vocabulary size", ylabel="Training tokens", title="Compression vs vocabulary")
        axes[1].bar(list(held), ratios); axes[1].set(ylabel="Bytes per token", title="Transfer to held-out text")
        axes[2].plot(sorted(byte_frequencies, reverse=True), marker=".", label="bytes / no merges")
        axes[2].plot(sorted(frequencies, reverse=True), marker=".", label="16 merges")
        axes[2].legend()
        axes[2].set(xlabel="Token frequency rank", ylabel="Count", title="Frequency distribution changes")
        fig.tight_layout()
        measurements = [{"requested_merges": n, "vocabulary": v, "training_tokens": c} for n, v, c in zip(budgets, vocab, counts)]
        failures = [{"case": name, "text": sample, "tokens_learned": len(encode(sample, learned)), "tokens_no_merges": len(encode(sample, [])), "roundtrip": decode(encode(sample, learned), learned) == sample} for name, sample in held.items()]
        return fig, measurements, failures

    return (experiment,)


@app.cell(hide_code=True)
def _(experiment, mo, run_experiment):
    mo.stop(not run_experiment.value, mo.md("Press **Run fresh experiment**. Nothing trains automatically."))
    figure, measurements, failure_gallery = experiment()
    figure
    return failure_gallery, figure, measurements


@app.cell(hide_code=True)
def _(failure_gallery, measurements, mo):
    mo.vstack([mo.md("### Measurements"), mo.ui.table(measurements, selection=None), mo.md("### Failure gallery"), mo.ui.table(failure_gallery, selection=None)])
    return


@app.cell(hide_code=True)
def _(
    check_results,
    device,
    failure_gallery,
    figure,
    io,
    json,
    measurements,
    mo,
    torch,
):
    result_payload = {"notebook": '01_tokenization', "device": str(device), "torch": str(torch.__version__), "dtype": "float32", "measurements": measurements, "failure_gallery": failure_gallery, "checks": check_results}
    plot_buffer = io.BytesIO()
    figure.savefig(plot_buffer, format="png", dpi=140, bbox_inches="tight")
    mo.hstack([
        mo.download(data=json.dumps(result_payload, indent=2).encode(), filename="results.json", label="Export results JSON"),
        mo.download(data=plot_buffer.getvalue(), filename="plots.png", label="Export three-panel figure"),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6 · Teach it back

    Edit and save this Markdown cell after the experiment.

    - Explain the mechanism in five plain-language sentences: **[write here]**
    - Work through one small numerical/text example: **[write here]**
    - What was wrong or incomplete in my initial explanation? **[write here]**
    - Which observation supports my revised explanation? **[write here]**
    - What did the ablation break, and why? **[write here]**
    - What remains unclear, and what experiment would distinguish my hypotheses? **[write here]**

    **Done when:** checks pass, all three plots have an interpretation, the failure gallery has been examined, and this explanation is revised. Passing checks alone is not completion.

    Download the notebook from Molab to keep your code and explanations; export buttons above save experimental evidence separately.
    """)
    return


if __name__ == "__main__":
    app.run()

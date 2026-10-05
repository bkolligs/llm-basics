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
    # 02 · IDs become vectors

    A token ID is an address, not a numerical meaning. Implement equivalent ways to retrieve learned vectors and inspect a tiny next-token model. Prerequisites: matrix multiplication and cosine similarity. The optimizer, cross-entropy loss, and training loop are supplied.

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

    Will token ID 6 be more similar to ID 5 than ID 0? Will animals with identical next-token distributions necessarily acquire identical vectors? What changes if targets are shuffled?

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

    return (
        check_suite,
        close,
        require,
        resolve_device,
        seeded_tensor,
        timed,
        to_numpy,
    )


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

    `one_hot(ids, vocab_size)` accepts integer IDs of any shape and appends a vocabulary axis. `embedding_lookup(weight, ids)` returns the selected rows while preserving autograd. `cosine_matrix(vectors)` returns all pairwise cosine similarities, with zero-vector similarities defined as zero.

    Replace each `raise NotImplementedError` in the next cell. Keep the function signatures. Allocate tensors on the input tensor’s device; the supplied fixtures already use the selected backend.
    """)
    return


@app.cell
def _():
    def one_hot(ids, vocab_size):
        """Long tensor [...] -> float32 tensor [..., vocab_size] on ids.device."""
        raise NotImplementedError("one_hot")

    def embedding_lookup(weight, ids):
        """weight [V,D], ids [...] -> [...,D], differentiable with respect to weight."""
        raise NotImplementedError("embedding_lookup")

    def cosine_matrix(vectors):
        """[N,D] -> [N,N]. Clamp norm denominator away from zero."""
        raise NotImplementedError("cosine_matrix")

    return cosine_matrix, embedding_lookup, one_hot


@app.cell(hide_code=True)
def _(
    close,
    cosine_matrix,
    device,
    embedding_lookup,
    one_hot,
    require,
    seeded_tensor,
    torch,
):
    def check_one_hot():
        ids = torch.tensor([[0, 2], [1, 0]], device=device)
        result = one_hot(ids, 3)
        close(result, torch.nn.functional.one_hot(ids, 3).float())
        require(result.device == ids.device, "Preserve device.")

    def check_lookup():
        weight = seeded_tensor((5, 4), device).requires_grad_()
        ids = torch.tensor([2, 2, 4], device=device)
        result = embedding_lookup(weight, ids)
        close(result, torch.nn.functional.embedding(ids, weight))
        close(result, one_hot(ids, 5) @ weight)
        result.sum().backward()
        require(weight.grad is not None, "Do not detach the lookup.")
        close(weight.grad[2], torch.full((4,), 2.0, device=device))
        close(weight.grad[0], torch.zeros(4, device=device))

    def check_cosine():
        vectors = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., 0.]], device=device)
        normalized = torch.nn.functional.normalize(vectors, dim=-1)
        close(cosine_matrix(vectors), normalized @ normalized.T)

    exercise_cases = [("One-hot values and device", check_one_hot), ("Lookup equivalence and repeated-ID gradients", check_lookup), ("Cosines including zero vectors", check_cosine)]
    return (exercise_cases,)


@app.cell(hide_code=True)
def _(check_suite, exercise_cases, mo):
    check_results = check_suite(exercise_cases)
    checks_passed = all(row["status"] == "PASS" for row in check_results)
    mo.vstack([mo.md("## 4 · Behavioral checks"), mo.ui.table(check_results, selection=None)])
    return check_results, checks_passed


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Hint 1': 'An ID chooses a row. Multiplication by a one-hot vector should produce exactly the same row.', 'Hint 2': 'Repeated IDs must accumulate gradients into the same row; indexing already supports this.', 'Hint 3': 'Normalize each vector by its length, then compare all normalized pairs. Handle the zero vector explicitly.'})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · Run, plot, and break

    **Ablation:** the supplied run shuffles target labels while keeping inputs fixed. Compare loss and explain why learned vectors reflect supervision. Edit the synthetic sentence frequencies and inspect geometry again; do not interpret this tiny space as general semantics.

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
def _(
    cosine_matrix,
    device,
    embedding_lookup,
    plt,
    seeded_tensor,
    timed,
    to_numpy,
    torch,
):
    def experiment():
        words = ["cat", "dog", "fox", "runs", "sleeps", "eats", "."]
        # Original synthetic sentences; deliberately skew cat frequency.
        sequences = [[a, v, 6] for a in [0, 0, 0, 1, 2] for v in [3, 4, 5]]
        x = torch.tensor([s[i] for s in sequences for i in range(2)], device=device)
        y = torch.tensor([s[i + 1] for s in sequences for i in range(2)], device=device)
        def train(shuffle):
            weight = (seeded_tensor((7, 8), device) * .2).requires_grad_()
            output = (seeded_tensor((8, 7), device, 9) * .2).requires_grad_()
            targets = y.clone()
            if shuffle:
                permutation = torch.randperm(len(y), generator=torch.Generator().manual_seed(13)).to(device)
                targets = y[permutation]
            optimizer = torch.optim.Adam([weight, output], lr=.05)
            losses = []
            for step in range(100):
                optimizer.zero_grad()
                logits = embedding_lookup(weight, x) @ output
                loss = torch.nn.functional.cross_entropy(logits, targets)
                loss.backward(); optimizer.step()
                losses.append(float(loss.detach().cpu()))
            return weight.detach(), output.detach(), losses
        (trained, output, losses), elapsed = timed(device, lambda: train(False))
        _, _, shuffled_losses = train(True)
        similarities = to_numpy(cosine_matrix(trained))
        norms = to_numpy(torch.linalg.vector_norm(trained, dim=-1))
        counts = to_numpy(torch.bincount(x, minlength=7))
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        axes[0].plot(losses, label="structured"); axes[0].plot(shuffled_losses, label="shuffled targets"); axes[0].legend(); axes[0].set(xlabel="Step", ylabel="Cross entropy", title="Context vs broken supervision")
        im = axes[1].imshow(similarities, vmin=-1, vmax=1, cmap="coolwarm"); axes[1].set_xticks(range(7), words, rotation=60); axes[1].set_yticks(range(7), words); axes[1].set_title("Learned cosine geometry"); fig.colorbar(im, ax=axes[1])
        axes[2].bar(words, norms)
        for i, count in enumerate(counts):
            axes[2].text(i, norms[i] + .08, f"n={int(count)}", ha="center", fontsize=8)
        axes[2].margins(y=.2)
        axes[2].tick_params(axis="x", labelrotation=45)
        axes[2].set(xlabel="Token (n = input frequency)", ylabel="Embedding norm", title="Frequency is not meaning")
        fig.tight_layout()
        predictions = to_numpy((trained @ output).argmax(-1)).astype(int)
        failures = [{"token": word, "predicted_next": words[predictions[i]], "limitation": "No left context beyond this token; '.' never appears as input." if i == 6 else "Several valid next words cannot be captured by argmax alone."} for i, word in enumerate(words)]
        metrics = [{"run": "structured", "final_loss": losses[-1], "seconds": elapsed}, {"run": "shuffled targets", "final_loss": shuffled_losses[-1], "seconds": None}]
        return fig, metrics, failures

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
    result_payload = {"notebook": '02_embeddings', "device": str(device), "torch": str(torch.__version__), "dtype": "float32", "measurements": measurements, "failure_gallery": failure_gallery, "checks": check_results}
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

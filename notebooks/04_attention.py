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
    # 04 · One query gathers context

    Implement scaled dot-product attention before adding heads or blocks. You receive projected Q, K, and V so the focus is score scaling, probabilities, and value aggregation. Prerequisites: matrix multiplication and softmax.

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

    What happens if every logit is increased by 1000? Should changing the final value affect query zero? What happens to entropy without scaling?

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

    return check_suite, close, require, resolve_device, seeded_tensor, to_numpy


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

    `stable_softmax` normalizes the last axis. For an entirely masked row (all `-inf`), define the output as zeros. `scaled_scores` accepts `Q [...,T,D]`, `K [...,S,D]`. `causal_mask(T,S,offset,device)` returns booleans where True means allowed: key index ≤ query index + offset. `attention` returns `(output, weights)` and supports an optional broadcastable boolean mask.

    Replace each `raise NotImplementedError` in the next cell. Keep the function signatures. Allocate tensors on the input tensor’s device; the supplied fixtures already use the selected backend.
    """)
    return


@app.cell
def _():
    def stable_softmax(logits):
        """Normalize final axis stably; all -inf row -> zeros, not NaNs."""
        raise NotImplementedError("stable_softmax")

    def scaled_scores(q, k):
        """Q [...,T,D], K [...,S,D] -> [...,T,S], scaled by sqrt(D)."""
        raise NotImplementedError("scaled_scores")

    def causal_mask(query_length, key_length, offset, device):
        """Return [T,S] bool mask; True means allowed, accounting for query offset."""
        raise NotImplementedError("causal_mask")

    def attention(q, k, v, mask=None):
        """Return output [...,T,Dv], weights [...,T,S]. Use your score/softmax functions."""
        raise NotImplementedError("attention")

    return attention, causal_mask, scaled_scores, stable_softmax


@app.cell(hide_code=True)
def _(
    attention,
    causal_mask,
    close,
    device,
    require,
    scaled_scores,
    seeded_tensor,
    stable_softmax,
    torch,
):
    def check_softmax():
        logits = torch.tensor([[1000., 1001., 999.], [-float("inf"), -float("inf"), -float("inf")]], device=device)
        result = stable_softmax(logits)
        close(result[0], torch.softmax(logits[0], -1))
        close(result[1], torch.zeros(3, device=device))
        require(torch.isfinite(result).all(), "No NaNs in masked rows.")

    def check_scores():
        q, k = seeded_tensor((2, 3, 4), device), seeded_tensor((2, 5, 4), device, 8)
        close(scaled_scores(q, k), torch.einsum("btd,bsd->bts", q, k) / 2)

    def check_mask():
        result = causal_mask(2, 4, 2, device)
        expected = torch.tensor([[True, True, True, False], [True, True, True, True]], device=device)
        require(result.dtype == torch.bool, "Mask must be boolean.")
        require(torch.equal(result, expected), "Account for query offset.")

    def check_attention():
        q = seeded_tensor((1, 4, 8), device).requires_grad_()
        k, v = seeded_tensor((1, 4, 8), device, 8), seeded_tensor((1, 4, 3), device, 9)
        mask = causal_mask(4, 4, 0, device)
        result, weights = attention(q, k, v, mask)
        close(result, torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask))
        close(weights.sum(-1), torch.ones((1, 4), device=device))
        require((weights[..., ~mask] == 0).all(), "Forbidden positions must have zero weight.")
        changed = v.clone(); changed[:, -1] += 100
        close(attention(q, k, changed, mask)[0][:, :-1], result[:, :-1], "Future values cannot affect the past.")
        result.sum().backward(); require(torch.isfinite(q.grad).all(), "Finite gradients required.")
        zero, zero_weights = attention(q.detach(), k, v, torch.zeros_like(mask))
        close(zero, torch.zeros_like(zero)); close(zero_weights, torch.zeros_like(zero_weights))

    exercise_cases = [("Stable and fully masked softmax", check_softmax), ("Scaled scores", check_scores), ("Offset causal mask", check_mask), ("Attention reference, isolation, gradients", check_attention)]
    return (exercise_cases,)


@app.cell(hide_code=True)
def _(check_suite, exercise_cases, mo):
    check_results = check_suite(exercise_cases)
    checks_passed = all(row["status"] == "PASS" for row in check_results)
    mo.vstack([mo.md("## 4 · Behavioral checks"), mo.ui.table(check_results, selection=None)])
    return check_results, checks_passed


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Hint 1': 'Subtracting a row constant preserves softmax. Decide what happens before subtracting negative infinity from itself.', 'Hint 2': 'The last Q/K dimension determines scale; the sequence length does not.', 'Hint 3': 'A mask removes keys from the probability distribution before value aggregation.'})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · Run, plot, and break

    **Ablations:** remove the causal mask and perturb the last value; separately remove score scaling as dimension grows. The supplied experiment shows both. Distinguish a numerical failure from an information-leakage failure.

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
    attention,
    causal_mask,
    device,
    plt,
    scaled_scores,
    seeded_tensor,
    stable_softmax,
    to_numpy,
):
    def experiment():
        q, k, v = seeded_tensor((12, 32), device), seeded_tensor((12, 32), device, 8), seeded_tensor((12, 6), device, 9)
        mask = causal_mask(12, 12, 0, device)
        output, weights = attention(q, k, v, mask)
        dims = [4, 16, 64, 256]
        entropies = {"scaled": [], "unscaled": []}
        for dim in dims:
            qd, kd = seeded_tensor((64, dim), device), seeded_tensor((64, dim), device, 11)
            for name, scores in [("scaled", scaled_scores(qd, kd)), ("unscaled", qd @ kd.T)]:
                probabilities = stable_softmax(scores)
                entropies[name].append(float((-(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1).mean()).cpu()))
        altered = v.clone(); altered[-1] += 20
        masked_change = (attention(q, k, altered, mask)[0] - output).norm(dim=-1)
        unmasked_change = (attention(q, k, altered)[0] - attention(q, k, v)[0]).norm(dim=-1)
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        axes[0].imshow(to_numpy(weights), cmap="viridis"); axes[0].set(xlabel="Key", ylabel="Query", title="Causal attention weights")
        for name, values in entropies.items(): axes[1].plot(dims, values, marker="o", label=name)
        axes[1].legend(); axes[1].set(xlabel="Head dimension", ylabel="Mean entropy (nats)", title="Scaling controls saturation")
        axes[2].plot(to_numpy(masked_change), label="causal"); axes[2].plot(to_numpy(unmasked_change), label="mask removed"); axes[2].legend(); axes[2].set(xlabel="Query position", ylabel="Output change norm", title="Future-token perturbation")
        fig.tight_layout()
        metrics = [{"dimension": d, "scaled_entropy": s, "unscaled_entropy": u} for d, s, u in zip(dims, entropies["scaled"], entropies["unscaled"])]
        failures = [{"case": "mask removed", "evidence": f"Earlier-query max change: {float(unmasked_change[:-1].max().cpu()):.4f}", "interpretation": "Future information leaks."}, {"case": "fully masked row", "evidence": "Covered by checks", "interpretation": "Explicit zero-row convention prevents NaNs."}, {"case": "no score scaling", "evidence": f"Entropy at D=256: {entropies['unscaled'][-1]:.4f}", "interpretation": "Inspect saturated probabilities."}]
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
    result_payload = {"notebook": '04_attention', "device": str(device), "torch": str(torch.__version__), "dtype": "float32", "measurements": measurements, "failure_gallery": failure_gallery, "checks": check_results}
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

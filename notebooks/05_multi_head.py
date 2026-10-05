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
    # 05 · Multiple attention subspaces

    Compose multi-head attention using PyTorch’s supplied single-head attention primitive. This isolates reshaping, independent head computations, and output mixing. Prerequisites: notebook 04 concepts, but no imports from its answers.

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

    Is reshaping [B,T,D] directly to [B,H,T,D/H] sufficient? Does dropping one head always affect only a contiguous output slice?

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

    Use `x [B,T,D]`, `H` heads, `Dh=D/H`. `split_heads` returns `[B,H,T,Dh]`; `combine_heads` reverses it. Projection matrices are `[D,D]` with the convention `x @ W` (not Linear’s transposed storage). `multi_head` projects Q/K/V, splits heads, applies supplied SDPA with a causal boolean mask, joins heads, and applies `Wo`. Optional `drop_head` zeros one head’s output before joining.

    Replace each `raise NotImplementedError` in the next cell. Keep the function signatures. Allocate tensors on the input tensor’s device; the supplied fixtures already use the selected backend.
    """)
    return


@app.cell
def _():
    def split_heads(x, num_heads):
        """[B,T,D] -> [B,H,T,D/H]; reject D not divisible by H with ValueError."""
        raise NotImplementedError("split_heads")

    def combine_heads(x):
        """[B,H,T,Dh] -> [B,T,H*Dh], preserving head/feature ordering."""
        raise NotImplementedError("combine_heads")

    def multi_head(x, wq, wk, wv, wo, num_heads, drop_head=None):
        """Causal self-attention [B,T,D] -> [B,T,D]; use functional SDPA, dropout_p=0."""
        raise NotImplementedError("multi_head")

    return combine_heads, multi_head, split_heads


@app.cell(hide_code=True)
def _(
    close,
    combine_heads,
    device,
    multi_head,
    require,
    seeded_tensor,
    split_heads,
    torch,
):
    def check_shapes():
        x = torch.arange(48, device=device, dtype=torch.float32).reshape(2, 3, 8)
        split = split_heads(x, 2)
        require(split.shape == (2, 2, 3, 4), "Expected [B,H,T,Dh].")
        close(split[:, 1, 2], x[:, 2, 4:])
        close(combine_heads(split), x)
        try: split_heads(x, 3)
        except ValueError: pass
        else: raise AssertionError("Reject nondivisible dimensions.")

    def check_reference():
        x = seeded_tensor((2, 4, 8), device)
        matrices = [seeded_tensor((8, 8), device, seed) * .2 for seed in range(10, 14)]
        module = torch.nn.MultiheadAttention(8, 2, dropout=0, batch_first=True, bias=False, device=device)
        with torch.no_grad():
            module.in_proj_weight.copy_(torch.cat([w.T for w in matrices[:3]], 0))
            module.out_proj.weight.copy_(matrices[3].T)
        blocked = torch.ones((4, 4), dtype=torch.bool, device=device).triu(1)
        reference, _ = module(x, x, x, attn_mask=blocked, need_weights=False)
        close(multi_head(x, *matrices, 2), reference)

    def check_isolation():
        x = seeded_tensor((1, 5, 8), device).requires_grad_()
        identity = torch.eye(8, device=device)
        original = multi_head(x, identity, identity, identity, identity, 2)
        altered = x.detach().clone(); altered[:, -1] += 30
        close(multi_head(altered, identity, identity, identity, identity, 2)[:, :-1], original[:, :-1])
        original.square().sum().backward(); require(torch.isfinite(x.grad).all(), "Finite gradients.")
        ablated = multi_head(x.detach(), identity, identity, identity, identity, 2, drop_head=0)
        close(ablated[..., :4], torch.zeros_like(ablated[..., :4]))
        close(ablated[..., 4:], original.detach()[..., 4:])

    exercise_cases = [("Head ordering and inverse", check_shapes), ("PyTorch multi-head reference", check_reference), ("Causality, gradients, head ablation", check_isolation)]
    return (exercise_cases,)


@app.cell(hide_code=True)
def _(check_suite, exercise_cases, mo):
    check_results = check_suite(exercise_cases)
    checks_passed = all(row["status"] == "PASS" for row in check_results)
    mo.vstack([mo.md("## 4 · Behavioral checks"), mo.ui.table(check_results, selection=None)])
    return check_results, checks_passed


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Hint 1': 'Reshape introduces a head axis; transpose moves it before time. Reshape alone does not do both.', 'Hint 2': 'The attention primitive sees each head’s feature dimension, not the full model width.', 'Hint 3': 'Zero a head before output mixing. With a nonidentity output projection, its effect can spread across all output dimensions.'})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · Run, plot, and break

    **Ablation:** remove each head separately. Keep total model width fixed when changing the number of heads. The weights are random: describe geometric differences without claiming semantic specialization.

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
def _(device, multi_head, plt, seeded_tensor, split_heads, to_numpy, torch):
    def experiment():
        x = seeded_tensor((1, 12, 16), device)
        matrices = [seeded_tensor((16, 16), device, s) * .2 for s in range(10, 14)]
        baseline = multi_head(x, *matrices, 4)
        changes = [float((multi_head(x, *matrices, 4, drop_head=h) - baseline).norm().cpu()) for h in range(4)]
        q, k = split_heads(x @ matrices[0], 4), split_heads(x @ matrices[1], 4)
        # Supplied diagnostic for inspecting one-head attention, not a solution to composition.
        scores = (q @ k.transpose(-2, -1)) / 2
        mask = torch.ones((12, 12), device=device, dtype=torch.bool).tril()
        probabilities = torch.softmax(scores.masked_fill(~mask, -float("inf")), -1)
        entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1)
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        axes[0].imshow(to_numpy(probabilities[0, 0]), cmap="viridis"); axes[0].set(xlabel="Key", ylabel="Query", title="Head 0 weights")
        axes[1].bar(range(4), changes); axes[1].set(xlabel="Removed head", ylabel="Output change norm", title="Head contribution ablation")
        for h in range(4): axes[2].plot(to_numpy(entropy[0, h]), label=f"head {h}")
        axes[2].legend(ncol=2, fontsize=8); axes[2].set(xlabel="Query position", ylabel="Entropy (nats)", title="Distinct random subspaces")
        fig.tight_layout()
        metrics = [{"head": h, "removal_change": changes[h], "mean_entropy": float(entropy[0, h].mean().cpu())} for h in range(4)]
        failures = [{"case": f"remove head {h}", "change": changes[h], "interpretation": "Random heads can differ; this does not establish learned linguistic specialization."} for h in range(4)]
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
    result_payload = {"notebook": '05_multi_head', "device": str(device), "torch": str(torch.__version__), "dtype": "float32", "measurements": measurements, "failure_gallery": failure_gallery, "checks": check_results}
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

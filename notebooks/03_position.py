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
    # 03 · Tokens acquire order

    Implement four positional mechanisms and inspect what they preserve. Attention without position is permutation-equivariant (permuting inputs permutes outputs); pooling can make the result invariant. These are different statements. Prerequisites: broadcasting, sine/cosine, dot products.

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

    What should a rotation preserve? Does an analytic positional formula imply good extrapolation? Can ALiBi replace a causal mask?

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

    Use even feature dimensions. Sinusoidal positions interleave sine/cosine with frequency `10000**(-2*i/dim)`. Learned positions select table rows. RoPE rotates adjacent pairs using the same frequencies and accepts `[T,D]` plus positions `[T]`. ALiBi returns `[H,T,T]` with `-slope[h] * abs(query-key)`; causality is a separate mask.

    Replace each `raise NotImplementedError` in the next cell. Keep the function signatures. Allocate tensors on the input tensor’s device; the supplied fixtures already use the selected backend.
    """)
    return


@app.cell
def _():
    def sinusoidal(positions, dim):
        """positions [T] -> float32 [T,dim], interleaved sin/cos on the same device."""
        raise NotImplementedError("sinusoidal")

    def learned_positions(table, positions):
        """table [max_length,D], integer positions [T] -> [T,D]; preserve gradients."""
        raise NotImplementedError("learned_positions")

    def apply_rope(x, positions):
        """x [T,D], positions [T] -> rotated [T,D]; rotate adjacent pairs."""
        raise NotImplementedError("apply_rope")

    def alibi_bias(length, slopes):
        """slopes [H] -> float32 [H,length,length] on slopes.device; unmasked."""
        raise NotImplementedError("alibi_bias")

    return alibi_bias, apply_rope, learned_positions, sinusoidal


@app.cell(hide_code=True)
def _(
    alibi_bias,
    apply_rope,
    close,
    device,
    learned_positions,
    require,
    seeded_tensor,
    sinusoidal,
    torch,
):
    def check_sinusoidal():
        positions = torch.tensor([0, 1, 7], device=device)
        result = sinusoidal(positions, 4)
        expected = torch.stack([positions.float().sin(), positions.float().cos(), (positions.float() * .01).sin(), (positions.float() * .01).cos()], -1)
        close(result, expected)

    def check_learned():
        table = seeded_tensor((8, 4), device).requires_grad_()
        indices = torch.tensor([0, 3, 3], device=device)
        output = learned_positions(table, indices)
        close(output, table[indices]); output.sum().backward()
        close(table.grad[3], torch.full((4,), 2., device=device))

    def check_rope():
        x = seeded_tensor((4, 8), device)
        positions = torch.arange(4, device=device)
        rotated = apply_rope(x, positions)
        close(rotated[0], x[0], "Position zero must be identity.")
        close(torch.linalg.vector_norm(rotated, dim=-1), torch.linalg.vector_norm(x, dim=-1), "Rotations preserve norms.")
        example = apply_rope(torch.tensor([[1., 0., 1., 0.]], device=device), torch.tensor([1], device=device))
        close(example, torch.tensor([[0.5403023, 0.8414710, 0.9999500, 0.0099998]], device=device))
        q = seeded_tensor((4, 8), device, 3)
        close(apply_rope(q, positions) @ rotated.T, apply_rope(q, positions + 5) @ apply_rope(x, positions + 5).T, "Joint position shifts preserve QK dots.")

    def check_alibi():
        slopes = torch.tensor([.5, 1.], device=device)
        bias = alibi_bias(3, slopes)
        close(bias[0], torch.tensor([[0., -.5, -1.], [-.5, 0., -.5], [-1., -.5, 0.]], device=device))
        require(bias.shape == (2, 3, 3), "One bias matrix per head.")
        close(bias[1], 2 * bias[0])

    exercise_cases = [("Sinusoidal frequencies", check_sinusoidal), ("Learned table gradients", check_learned), ("RoPE rotations and relative shifts", check_rope), ("ALiBi distances", check_alibi)]
    return (exercise_cases,)


@app.cell(hide_code=True)
def _(check_suite, exercise_cases, mo):
    check_results = check_suite(exercise_cases)
    checks_passed = all(row["status"] == "PASS" for row in check_results)
    mo.vstack([mo.md("## 4 · Behavioral checks"), mo.ui.table(check_results, selection=None)])
    return check_results, checks_passed


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Hint 1': 'Position and feature index have different axes. Work through position zero before broadcasting.', 'Hint 2': 'A two-dimensional rotation mixes each adjacent pair; the frequency changes between pairs.', 'Hint 3': 'ALiBi changes scores, not values. A distance bias does not prohibit looking into the future.'})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · Run, plot, and break

    **Ablation:** compare RoPE with unchanged vectors and ALiBi with zero bias. The learned table intentionally ends at position 15. Analyze out-of-range cases on CPU before requesting unsupported indices on CUDA: the experiment checks table capacity before lookup.

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
    alibi_bias,
    apply_rope,
    device,
    learned_positions,
    plt,
    seeded_tensor,
    sinusoidal,
    to_numpy,
    torch,
):
    def experiment():
        positions = torch.arange(64, device=device)
        encoding = sinusoidal(positions, 16)
        vector = seeded_tensor((1, 16), device).expand(64, -1)
        rotated = apply_rope(vector, positions)
        scores = rotated[0] @ rotated.T / 4
        biases = alibi_bias(64, torch.tensor([.05, .2], device=device))
        query_scores = seeded_tensor((1, 64), device, 2).squeeze(0)
        with_bias = torch.softmax(query_scores + biases[1, -1], -1)
        no_bias = torch.softmax(query_scores, -1)
        table = seeded_tensor((16, 16), device, 4)
        failures = []
        for length in [8, 16, 32, 64]:
            try:
                if length > table.shape[0]:
                    raise IndexError("Outside learned position table")
                learned_positions(table, torch.arange(length, device=device))
                outcome = "Representable by supplied table"
            except (IndexError, RuntimeError):
                outcome = "Outside learned table; extending indices is not trained extrapolation"
            failures.append({"length": length, "learned_table": outcome, "analytic_methods": "Defined, but quality at this length is unproven"})
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        axes[0].imshow(to_numpy(encoding), aspect="auto", cmap="coolwarm"); axes[0].axhline(15.5, color="black", linestyle="--"); axes[0].set(xlabel="Feature", ylabel="Position", title="Sinusoids beyond nominal length 16")
        axes[1].plot(to_numpy(scores), label="RoPE"); axes[1].axhline(float((vector[0] @ vector[0] / 4).cpu()), linestyle="--", label="no position"); axes[1].legend(); axes[1].set(xlabel="Relative distance", ylabel="QK score", title="Position changes similarity")
        axes[2].plot(to_numpy(no_bias), label="no distance bias"); axes[2].plot(to_numpy(with_bias), label="ALiBi"); axes[2].legend(); axes[2].set(xlabel="Key position (query=63)", ylabel="Attention probability", title="Distance prior")
        fig.tight_layout()
        metrics = [{"quantity": "RoPE norm max error", "value": float((rotated.norm(dim=-1)-vector.norm(dim=-1)).abs().max().cpu())}, {"quantity": "ALiBi mass on last 8 keys", "value": float(with_bias[-8:].sum().cpu())}, {"quantity": "No-bias mass on last 8 keys", "value": float(no_bias[-8:].sum().cpu())}]
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
    result_payload = {"notebook": '03_position', "device": str(device), "torch": str(torch.__version__), "dtype": "float32", "measurements": measurements, "failure_gallery": failure_gallery, "checks": check_results}
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

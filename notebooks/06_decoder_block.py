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
    # 06 · Assemble a decoder block

    Build a pre-norm decoder block and train it on one tiny synthetic batch. Attention, embeddings, output head, data, and optimizer are supplied. Focus on residual paths, RMSNorm, SwiGLU, and composition. Prerequisites: gradients and residual connections.

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

    If both sublayers output zeros, what should the block return? Does RMSNorm force mean zero? Does fitting one batch demonstrate language understanding?

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

    return check_suite, close, require, resolve_device, seeded_tensor, timed


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

    `rms_norm(x, weight, eps)` rescales the final dimension without subtracting its mean. `swiglu(x, Wg, Wu, Wd)` uses `SiLU(x@Wg) * (x@Wu)` followed by down projection. `decoder_block(x, attention_fn, parameters, ...)` uses pre-norm attention and pre-norm feed-forward sublayers, each with a residual addition. Switches independently disable normalization and residual paths for experiments. The supplied attention callable is already causal.

    Replace each `raise NotImplementedError` in the next cell. Keep the function signatures. Allocate tensors on the input tensor’s device; the supplied fixtures already use the selected backend.
    """)
    return


@app.cell
def _():
    def rms_norm(x, weight, eps=1e-5):
        """x [...,D], weight [D] -> [...,D]; normalize by root mean square."""
        raise NotImplementedError("rms_norm")

    def swiglu(x, w_gate, w_up, w_down):
        """x [...,D], gate/up [D,F], down [F,D] -> [...,D]."""
        raise NotImplementedError("swiglu")

    def decoder_block(x, attention_fn, parameters, use_norm=True, use_residual=True):
        """parameters: norm1/norm2 [D], gate/up [D,F], down [F,D]. Pre-norm, two residual paths."""
        raise NotImplementedError("decoder_block")

    return decoder_block, rms_norm, swiglu


@app.cell(hide_code=True)
def _(
    close,
    decoder_block,
    device,
    require,
    rms_norm,
    seeded_tensor,
    swiglu,
    torch,
):
    def check_norm():
        x = seeded_tensor((2, 3, 8), device).requires_grad_()
        scale = seeded_tensor((8,), device, 4)
        close(rms_norm(x, scale), torch.nn.functional.rms_norm(x, (8,), scale, eps=1e-5))
        close(rms_norm(torch.zeros_like(x), scale), torch.zeros_like(x))
        rms_norm(x, scale).sum().backward(); require(torch.isfinite(x.grad).all(), "Finite norm gradients.")

    def check_swiglu():
        x = seeded_tensor((2, 3, 4), device)
        gate, up, down = seeded_tensor((4, 6), device), seeded_tensor((4, 6), device, 8), seeded_tensor((6, 4), device, 9)
        expected = torch.nn.functional.silu(x @ gate) * (x @ up)
        close(swiglu(x, gate, up, down), torch.nn.functional.linear(expected, down.T))

    def check_block():
        x = seeded_tensor((1, 3, 4), device).requires_grad_()
        params = {"norm1": torch.ones(4, device=device), "norm2": torch.ones(4, device=device), "gate": torch.zeros((4, 6), device=device), "up": torch.zeros((4, 6), device=device), "down": torch.zeros((6, 4), device=device)}
        zero_attention = lambda z: torch.zeros_like(z)
        close(decoder_block(x, zero_attention, params), x, "Zero sublayers preserve the residual stream.")
        close(decoder_block(x, zero_attention, params, use_residual=False), torch.zeros_like(x))
        seen = []
        def spy(z):
            seen.append(z)
            return z * .1
        output = decoder_block(x, spy, params)
        close(seen[0], rms_norm(x, params["norm1"]), "Normalize before attention.")
        close(output, x + .1 * rms_norm(x, params["norm1"]))
        output.sum().backward(); require(torch.isfinite(x.grad).all(), "Finite block gradients.")
        # Nonzero FFN catches incorrect second residual or normalization placement.
        nonzero = dict(params)
        nonzero.update({"gate": seeded_tensor((4, 6), device, 20)*.2, "up": seeded_tensor((4, 6), device, 21)*.2, "down": seeded_tensor((6, 4), device, 22)*.2})
        intermediate = x.detach() + .1 * rms_norm(x.detach(), nonzero["norm1"])
        expected = intermediate + swiglu(rms_norm(intermediate, nonzero["norm2"]), nonzero["gate"], nonzero["up"], nonzero["down"])
        close(decoder_block(x.detach(), lambda z: .1*z, nonzero), expected)
        raw_mid = x.detach() * 1.1
        close(decoder_block(x.detach(), lambda z: .1*z, nonzero, use_norm=False), raw_mid + swiglu(raw_mid, nonzero["gate"], nonzero["up"], nonzero["down"]))

    exercise_cases = [("RMSNorm reference, zeros, gradients", check_norm), ("SwiGLU gated projection", check_swiglu), ("Residual identity, pre-norm, composition", check_block)]
    return (exercise_cases,)


@app.cell(hide_code=True)
def _(check_suite, exercise_cases, mo):
    check_results = check_suite(exercise_cases)
    checks_passed = all(row["status"] == "PASS" for row in check_results)
    mo.vstack([mo.md("## 4 · Behavioral checks"), mo.ui.table(check_results, selection=None)])
    return check_results, checks_passed


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'Hint 1': 'RMS normalization changes scale but does not center the features. Inspect a constant nonzero input.', 'Hint 2': 'SwiGLU has two upward projections: one gates the other before the downward projection.', 'Hint 3': 'For pre-norm, normalization belongs inside each residual branch. The second branch receives the updated residual stream.'})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · Run, plot, and break

    **Ablations:** compare the same initialization with normalization removed and with residual paths removed. All may fit this tiny task; the magnitude and stability of gradients still differ. Reverse the sequence order to expose memorization. A causal perturbation metric checks that future tokens cannot change earlier logits.

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
def _(decoder_block, device, plt, timed, torch):
    def experiment():
        # Supplied prerequisite model/training driver. No dependency on previous lab answers.
        class TinyDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(12, 16)
                self.position = torch.nn.Embedding(8, 16)
                self.attention = torch.nn.MultiheadAttention(16, 2, batch_first=True, dropout=0)
                self.output = torch.nn.Linear(16, 12)
                self.parts = torch.nn.ParameterDict({"norm1": torch.nn.Parameter(torch.ones(16)), "norm2": torch.nn.Parameter(torch.ones(16)), "gate": torch.nn.Parameter(torch.randn(16, 32)*.1), "up": torch.nn.Parameter(torch.randn(16, 32)*.1), "down": torch.nn.Parameter(torch.randn(32, 16)*.1)})
            def forward(self, ids, use_norm, use_residual):
                x = self.embedding(ids) + self.position(torch.arange(ids.shape[1], device=ids.device))
                def attend(z):
                    blocked = torch.ones((z.shape[1], z.shape[1]), dtype=torch.bool, device=z.device).triu(1)
                    return self.attention(z, z, z, attn_mask=blocked, need_weights=False)[0]
                hidden = decoder_block(x, attend, self.parts, use_norm=use_norm, use_residual=use_residual)
                return self.output(hidden), hidden
        sequences = torch.tensor([[0,1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7,8,9], [2,3,4,5,6,7,8,9,10], [3,4,5,6,7,8,9,10,11]], device=device)
        inputs, targets = sequences[:, :-1], sequences[:, 1:]
        traces = {}; failures = []; metrics = []
        for label, use_norm, use_residual in [("baseline", True, True), ("no norm", False, True), ("no residual", True, False)]:
            # Initialize on CPU from the same seed for each ablation.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                model = TinyDecoder().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=.02)
            losses, gradients, activations = [], [], []
            def train():
                for step in range(80):
                    optimizer.zero_grad()
                    logits, hidden = model(inputs, use_norm, use_residual)
                    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 12), targets.reshape(-1))
                    if not torch.isfinite(loss):
                        failures.append({"case": label, "evidence": f"Nonfinite loss at step {step}"}); break
                    loss.backward()
                    grad_norm = sum(float(p.grad.detach().square().sum().cpu()) for p in model.parameters() if p.grad is not None) ** .5
                    losses.append(float(loss.detach().cpu())); gradients.append(grad_norm); activations.append(float(hidden.detach().square().mean().sqrt().cpu()))
                    optimizer.step()
            _, elapsed = timed(device, train)
            traces[label] = (losses, gradients, activations)
            with torch.no_grad():
                logits, _ = model(inputs, use_norm, use_residual)
                accuracy = float((logits.argmax(-1) == targets).float().mean().cpu())
                changed = inputs.clone(); changed[:, -1] = (changed[:, -1]+3) % 12
                perturb_logits, _ = model(changed, use_norm, use_residual)
                leakage = float((perturb_logits[:, :-1] - logits[:, :-1]).abs().max().cpu())
                # Deliberate distribution shift: reverse the training sequences.
                reversed_seq = sequences.flip(-1)
                shifted_logits, _ = model(reversed_seq[:, :-1], use_norm, use_residual)
                shifted_accuracy = float((shifted_logits.argmax(-1) == reversed_seq[:, 1:]).float().mean().cpu())
            metrics.append({"run": label, "final_loss": losses[-1] if losses else None, "train_accuracy": accuracy, "reversed_accuracy": shifted_accuracy, "future_leakage": leakage, "seconds": elapsed})
            failures.append({"case": label + " / reversed order", "evidence": f"train accuracy {accuracy:.3f}, reversed accuracy {shifted_accuracy:.3f}; memorization does not imply generalization"})
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        for label, trace in traces.items():
            for axis, values in zip(axes, trace): axis.plot(values, label=label)
        for axis, title, ylabel in zip(axes, ["Tiny-batch overfit", "Gradient transport", "Residual-stream scale"], ["Cross entropy", "Gradient L2 norm", "Activation RMS"]):
            axis.set(xlabel="Step", ylabel=ylabel, title=title); axis.legend()
        fig.tight_layout()
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
    result_payload = {"notebook": '06_decoder_block', "device": str(device), "torch": str(torch.__version__), "dtype": "float32", "measurements": measurements, "failure_gallery": failure_gallery, "checks": check_results}
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

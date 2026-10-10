# laya-next

An independent experimental fork of [LAYA](https://github.com/NandhaKishorM/laya),
updated through upstream 0.4.2. The upstream license and attribution are preserved.
This fork is not an official upstream release.

## Added features

- [Residual encoder adapters](docs/residual-adapters.md): zero-initialized parallel
  bottleneck branches for selected ModernBERT layers, checkpoint loading, and an
  ablation switch. Training is required; attaching adapters is not a quality upgrade.
- [Numeric features](docs/numeric-features.md): exact decimal digit/place features
  and alignment to tokenizer offsets. These utilities do not automatically modify
  the decision model or its predictions.
- [Comparison specialist](docs/comparison.md): an explicit opt-in wrapper combining
  learned operator/operand binding with exact unit comparisons. It requires a
  separately trained local bundle; no specialist weights are included or downloaded.

Default prediction behavior is retained. The adapter path supports eager PyTorch;
unsupported acceleration/export paths reject adapted checkpoints rather than silently
dropping the branches. See each feature's documentation for limits and runnable checks.

## Compatibility updates

The residual adapters integrate with the shared trainer, including a frozen encoder
with trainable adapters, and the supported parallel-option layout. Unsupported
backend requests are rejected or explicitly fall back to eager execution; they do
not silently omit trained adapter weights.

The TypeScript SDK accepts and validates the optional HTTP `x_jev_confidence` field
on choice and score answers. Older responses without that field remain valid.

## Install this fork

Use a separate virtual environment to avoid replacing another LAYA installation:

```bash
git clone https://github.com/debjyotikm/laya-next.git
cd laya-next
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The import remains `laya`. Select a PyTorch build appropriate for your hardware when
using CUDA. Upstream PyPI packages and hosted documentation do not include these additions.

## Validation and scope

The repository includes synthetic unit/regression tests and a synthetic adapter
benchmark. They check engineering behavior, not general decision-quality improvement.
No new accuracy claim is made by this fork; evaluate trained models on your own held-out
tasks and measure latency and memory on your hardware.

The additions contain reusable code, tests, and documentation, not experiment logs,
training corpora, checkpoints, or infrastructure configuration. Future contributions
should retain that boundary and follow [AGENTS.md](AGENTS.md) and
[CONTRIBUTING.md](CONTRIBUTING.md).

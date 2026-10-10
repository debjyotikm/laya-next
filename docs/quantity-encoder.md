# Whole-quantity encoder

`laya.quantity` is an optional trainable component built on the
[numeric features](numeric-features.md). It combines digit, decimal-place and sign
embeddings **before** pooling each complete quantity. The resulting vector is
broadcast to tokens containing that quantity's digits.

It does not change an `Agent` automatically. It provides no pretrained weights,
unit conversion, operand binding, arithmetic solver or accuracy guarantee.
Callers own integration with a model and must train the new parameters.

## Prepare features without changing tokenization

Use complete numeric-literal spans and the original tokenizer's character offsets.
The following small example uses explicit offsets for three synthetic tokens:

```python
from laya.numeric import align_slots
from laya.quantity import QuantityEncoder, prepare_quantities

text = "limit: 125"
offsets = [(0, 7), (7, 8), (8, 10)]
digits = align_slots(text, [(7, 10)], offsets)
batch = prepare_quantities([digits], sequence_length=3)
encoder = QuantityEncoder(rank=16, width=32)
features = encoder(batch)  # shape: (1, 3, 16)
```

In a real model, retain its token IDs and offsets. Do not silently decode and
retokenize a different string. Padding receives zero only when it is excluded
from the supplied digit metadata. `align_slots` rejects incomplete literals and
invalid offsets. `prepare_quantities` copies rows into immutable tuples and
rejects inconsistent signs, duplicate powers, or out-of-grid token positions.

Each quantity contributes once per destination token, regardless of how many of
its digits occupy that token. A token shared by multiple quantities receives
their mean. Tokens without assigned digits receive zero. This invariance applies
to the **side vector**, not to the full contextual model.

The module has 14,816 parameters with output rank 256, width 32 and the default
decimal-power range. This count does not describe inference or training memory.

## Initialization and training

The output projection is randomly initialized by default. If adding the output
directly as a residual, use `zero_output=True` to preserve the initial function.
If an enclosing adapter already has a zero-initialized output projection, the
default avoids adding a second zero projection that can block learning.

Capture the immutable `QuantityBatch` belonging to each forward when using
activation checkpointing. Do not use a mutable global or a "last batch" cache.
The component supports gradients through all its learned representations;
metadata is input-only and contains no labels. CPU FP32 and BF16/autocast behavior
is tested. CUDA operations are not separately qualified by these CPU tests.

The supported power range defaults to -32 through +32. Out-of-range powers raise;
unseen powers inside that range still have untrained embeddings. Leading and
trailing zeros are retained, so equivalent numeric spellings can yield different
representations. There is no automatic numerical extrapolation guarantee.

## Save and reload

Save `encoder.get_config()` as JSON and the module's `state_dict()` as a separate
weights file. Reconstruct using `QuantityEncoder.from_config(config)` and load
with `load_state_dict(state, strict=True)`. Use a tensor-only format or
`torch.load(path, weights_only=True)` for a PyTorch weights file.

This is a component checkpoint, **not** a complete LAYA checkpoint accepted by
`Agent`. It does not include optimizer state or change LAYA's model configuration.
A complete architecture integration needs its own versioned loading contract.

## Checks and limitations

```bash
python -m pytest tests/test_quantity.py -q
python tests/test_hooks_api.py
```

The tests use synthetic CPU fixtures, including mixed lengths, empty rows, shared
tokens, explicit Unicode offsets, strict reload, zero-init learning and separate
metadata across checkpointed forwards. They establish component behavior, not
improved task accuracy, end-to-end encoder integration, latency or GPU performance.

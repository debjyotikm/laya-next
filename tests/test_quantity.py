"""Synthetic CPU checks for the optional whole-quantity component."""
import copy
import json

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from laya.numeric import AlignedDigit, align_slots
from laya.quantity import QuantityBatch, QuantityEncoder, prepare_quantities


def digits(text="123", tokens=(1, 1, 1), quantity=0):
    return tuple(AlignedDigit(quantity, token, int(value), len(text) - index - 1, False)
                 for index, (value, token) in enumerate(zip(text, tokens)))


def encoder(**kwargs):
    torch.manual_seed(17)
    return QuantityEncoder(4, width=8, **kwargs)


def test_pooling_and_broadcast_ignore_digit_token_segmentation():
    model = encoder()
    whole = model(prepare_quantities([digits()], 5))
    split = model(prepare_quantities([digits(tokens=(1, 2, 3))], 5))
    for token in (1, 2, 3):
        torch.testing.assert_close(whole[0, 1], split[0, token], rtol=0, atol=0)
    assert torch.count_nonzero(whole[:, (0, 2, 3, 4)]) == 0
    assert torch.count_nonzero(split[:, (0, 4)]) == 0


def test_shared_token_averages_quantities_once_each():
    model = encoder()
    a = digits()
    b = digits("987", quantity=9)
    one = model(prepare_quantities([a], 3))[0, 1]
    two = model(prepare_quantities([b], 3))[0, 1]
    combined = model(prepare_quantities([a + b], 3))[0, 1]
    torch.testing.assert_close(combined, (one + two) / 2)


def test_batching_empty_rows_and_padding():
    model = encoder()
    row = digits()
    single = model(prepare_quantities([row], 3))
    batch = model(prepare_quantities([(), row, digits("987")], 6))
    torch.testing.assert_close(batch[1, :3], single[0])
    assert torch.count_nonzero(batch[0]) == 0
    assert torch.count_nonzero(batch[:, 3:]) == 0
    assert torch.count_nonzero(model(prepare_quantities([(), ()], 4))) == 0


def test_sign_place_and_digit_are_learned_features():
    model = encoder()
    original = digits()
    negatives = tuple(AlignedDigit(s.quantity, s.token, s.digit, s.power, True) for s in original)
    shifted = tuple(AlignedDigit(s.quantity, s.token, s.digit, s.power + 1, False) for s in original)
    permuted = tuple(AlignedDigit(s.quantity, s.token, s.digit, -s.power, False) for s in original)
    values = model(prepare_quantities([original, negatives, shifted, permuted, digits("987")], 3))[:, 1]
    assert all(not torch.equal(values[0], value) for value in values[1:])
    values.square().sum().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_explicit_text_offsets_keep_unicode_sign_and_scientific_power():
    text = "μ=-1.25e+3"
    slots = align_slots(text, [(2, len(text))], [(0, 2), (2, len(text))])
    assert [slot.power for slot in slots] == [3, 2, 1]
    assert all(slot.negative for slot in slots)
    value = encoder()(prepare_quantities([slots], 4))
    assert torch.isfinite(value).all()
    assert torch.count_nonzero(value[:, (0, 2, 3)]) == 0


def test_input_sequences_are_frozen():
    rows = [list(digits())]
    batch = prepare_quantities(rows, 3)
    rows[0].clear()
    assert len(batch.rows[0]) == 3
    with pytest.raises(AttributeError):
        batch.sequence_length = 9


@pytest.mark.parametrize("row", [
    [object()], [AlignedDigit(0, -1, 1, 0, False)], [AlignedDigit(0, 3, 1, 0, False)],
    [AlignedDigit(-1, 0, 1, 0, False)], [AlignedDigit(0, 0, 10, 0, False)],
    [AlignedDigit(True, 0, 1, 0, False)], [AlignedDigit(0, 0, 1, 0.5, False)],
    [AlignedDigit(0, 0, 1, 0, 1)], list(digits()) + [digits()[0]],
    [AlignedDigit(0, 0, 1, 0, False), AlignedDigit(0, 1, 2, 1, True)],
])
def test_reject_invalid_digits(row):
    with pytest.raises(ValueError):
        prepare_quantities([row], 3)


@pytest.mark.parametrize("rows,length", [([], 3), ([()], 0), ([()], True), ("123", 3), (["123"], 3)])
def test_reject_invalid_grid(rows, length):
    with pytest.raises(ValueError):
        prepare_quantities(rows, length)


def test_direct_metadata_constructor_checks_immutability():
    with pytest.raises(ValueError):
        QuantityBatch([digits()], 3)
    with pytest.raises(ValueError):
        QuantityBatch((list(digits()),), 3)


@pytest.mark.parametrize("kwargs", [{"rank": True}, {"rank": 0}, {"width": 0},
                                   {"min_power": 2, "max_power": 1}, {"zero_output": 1}])
def test_reject_invalid_encoder_config(kwargs):
    with pytest.raises(ValueError):
        QuantityEncoder(**dict({"rank": 4}, **kwargs))


def test_range_and_type_errors():
    model = encoder(min_power=-1, max_power=1)
    with pytest.raises(ValueError, match="power"):
        model(prepare_quantities([digits()], 3))
    with pytest.raises(TypeError, match="QuantityBatch"):
        model([[1, 2, 3]])


def test_versioned_config_and_strict_weight_roundtrip(tmp_path):
    model = encoder(zero_output=True)
    with torch.no_grad():
        model.project.weight.fill_(0.1)
    config = json.loads(json.dumps(model.get_config()))
    loaded = QuantityEncoder.from_config(config)
    path = tmp_path / "quantity.pt"
    torch.save(model.state_dict(), path)
    loaded.load_state_dict(torch.load(path, weights_only=True), strict=True)
    batch = prepare_quantities([digits()], 3)
    torch.testing.assert_close(model(batch), loaded(batch), rtol=0, atol=0)
    for bad in ({**config, "version": True}, {**config, "version": 2},
                {**config, "unknown": 0}, {k: v for k, v in config.items() if k != "rank"}):
        with pytest.raises(ValueError):
            QuantityEncoder.from_config(bad)
    assert sum(p.numel() for p in QuantityEncoder(256).parameters()) == 14816


def test_zero_initialization_then_two_step_learning():
    model = encoder(zero_output=True)
    batch = prepare_quantities([digits()], 3)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    output = model(batch)
    assert torch.count_nonzero(output) == 0
    (output[:, 1] - 1).square().sum().backward()
    assert model.project.weight.grad.abs().sum() > 0
    assert torch.count_nonzero(model.digit.weight.grad) == 0
    opt.step()
    opt.zero_grad()
    (model(batch)[:, 1] - 1).square().sum().backward()
    assert model.digit.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("stored_bfloat16", [False, True])
def test_cpu_bfloat16_and_autocast(stored_bfloat16):
    model = encoder()
    if stored_bfloat16:
        model = model.to(torch.bfloat16)
    batch = prepare_quantities([digits(), ()], 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(batch)
        loss = output.float().square().sum()
    assert output.dtype == model.project.weight.dtype
    assert torch.isfinite(output).all()
    loss.backward()
    assert torch.isfinite(model.project.weight.grad).all()


@pytest.mark.parametrize("reentrant", [False, True])
def test_checkpoint_replay_keeps_each_forwards_metadata(reentrant):
    direct = encoder()
    replay = copy.deepcopy(direct)
    batches = [prepare_quantities([digits()], 3), prepare_quantities([digits("987")], 5)]

    def gradients(model, use_checkpoint):
        outputs = []
        for batch in batches:
            hidden = torch.zeros(*batch.shape, model.rank, requires_grad=True)

            def forward(value, metadata=batch):
                return value + model(metadata)

            outputs.append(checkpoint(forward, hidden, use_reentrant=reentrant)
                           if use_checkpoint else forward(hidden))
        with torch.no_grad():
            model(prepare_quantities([digits("456")], 4))
        sum(value.square().sum() for value in outputs).backward()
        return outputs, [p.grad for p in model.parameters()]

    expected, expected_grads = gradients(direct, False)
    actual, actual_grads = gradients(replay, True)
    for first, second in zip(expected + expected_grads, actual + actual_grads):
        torch.testing.assert_close(first, second, rtol=0, atol=0)

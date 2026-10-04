"""Offline residual architecture contracts; tiny encoders and authored inputs only."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

from laya.agent import Agent
from laya.common import DecisionModel, build_model
from laya._compile import compile_model
from laya._residual import ResidualAdapters


CONFIG = {"version": 1, "layers": [0, 1], "rank": 8, "scale": 2.0}


def model_config():
    return ModernBertConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=2,
                            max_position_embeddings=128, local_attention=16,
                            pad_token_id=0, cls_token_id=1, sep_token_id=2,
                            reference_compile=False, attention_dropout=0.0,
                            mlp_dropout=0.0, embedding_dropout=0.0)


def make_model(config=CONFIG, head_layers=1):
    encoder_config = model_config()
    encoder_config._attn_implementation = "sdpa"
    return DecisionModel(ModernBertModel(encoder_config), head_layers=head_layers,
                         dropout=0.0, residual_adapters=copy.deepcopy(config))


def inputs(rows=2, length=12):
    ids = torch.arange(rows * length).reshape(rows, length) % 59 + 4
    mask = torch.ones_like(ids)
    mask[0, -2:] = 0
    return (ids, mask, torch.tensor([[2, 5]]).expand(rows, -1),
            torch.ones(rows, 2, dtype=torch.bool), torch.arange(rows) % 3)


def nonzero_branches(model):
    with torch.no_grad():
        for branch in model.residual_adapters.branches.values():
            branch.up.weight.normal_(std=0.03)


def bundle(tmp_path, model=None):
    model = make_model() if model is None else model
    cfg = {"encoder": "local-test-encoder", "head_layers": 1, "act_costs": {"escalate": 1},
           "max_len": 96, "head_max_len": 64, "temperature": [1.0, 1.0, 1.0],
           "residual_adapters": copy.deepcopy(CONFIG)}
    (tmp_path / "rl_agent_config.json").write_text(json.dumps(cfg))
    model.encoder.config.save_pretrained(tmp_path / "encoder")
    vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4,
             "yes": 5, "no": 6, "Select": 7, "item": 8}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", cls_token="[CLS]",
                            sep_token="[SEP]", mask_token="[MASK]", unk_token="[UNK]").save_pretrained(
                                tmp_path / "tokenizer")
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    return cfg


@pytest.mark.parametrize("head_layers", [0, 1])
def test_zero_initialization_and_stock_keys(head_layers):
    torch.manual_seed(41)
    stock = make_model(None, head_layers).eval()
    candidate = DecisionModel(copy.deepcopy(stock.encoder), head_layers=head_layers,
                              dropout=0.0, residual_adapters=CONFIG).eval()
    missing, unexpected = candidate.load_state_dict(stock.state_dict(), strict=False)
    assert missing and all(key.startswith("residual_adapters.") for key in missing)
    assert not unexpected
    assert not any(key.startswith("residual_adapters.") for key in stock.state_dict())
    with torch.no_grad():
        for expected, actual in zip(stock(*inputs()), candidate(*inputs())):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert sum(p.numel() for p in candidate.residual_adapters.parameters()) == 576


def test_gradients_freezing_and_nonzero_effect():
    model = make_model().train()
    model.encoder.requires_grad_(False)
    optimizer = torch.optim.SGD(model.residual_adapters.parameters(), lr=0.1)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(*inputs())[0], torch.tensor([0, 1]))
        loss.backward()
        assert all(p.grad is None for p in model.encoder.parameters())
        assert all(b.up.weight.grad.abs().sum() > 0 for b in model.residual_adapters.branches.values())
        optimizer.step()
    assert all(b.down.weight.grad.abs().sum() > 0 for b in model.residual_adapters.branches.values())
    model.eval()
    with torch.no_grad():
        active = model(*inputs())[0]
        model.set_residual_enabled(False)
        bypassed = model(*inputs())[0]
        model.set_residual_enabled(True)
        torch.testing.assert_close(model(*inputs())[0], active)
    assert not torch.equal(active, bypassed)


def test_deepcopy_hooks_belong_to_copy():
    original = make_model().eval()
    clone = copy.deepcopy(original)
    nonzero_branches(clone)
    assert all(torch.count_nonzero(b.up.weight) == 0 for b in original.residual_adapters.branches.values())
    with torch.no_grad():
        assert not torch.equal(original(*inputs())[0], clone(*inputs())[0])


@pytest.mark.parametrize("freeze_encoder", [False, True])
def test_checkpointed_encoder_matches_gradients(freeze_encoder):
    ordinary = make_model().train()
    nonzero_branches(ordinary)
    ordinary.encoder.requires_grad_(not freeze_encoder)
    checkpointed = copy.deepcopy(ordinary)
    checkpointed.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for model in (ordinary, checkpointed):
        model(*inputs())[0].square().sum().backward()
    for (name, param), (other_name, other) in zip(ordinary.named_parameters(), checkpointed.named_parameters()):
        assert name == other_name
        if param.grad is None:
            assert other.grad is None
        else:
            torch.testing.assert_close(param.grad, other.grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("changes", [{"version": 2}, {"version": True}, {"layers": []},
                                      {"layers": [0, 0]}, {"layers": [2]}, {"layers": [True]},
                                      {"rank": 0}, {"rank": True}, {"scale": float("nan")},
                                      {"scale": float("inf")}, {"scale": 0}, {"extra": 1}])
def test_rejects_invalid_config_without_attaching(changes):
    encoder = make_model(None).encoder
    with pytest.raises(ValueError):
        ResidualAdapters(encoder, dict(CONFIG, **changes))
    assert all(not layer._forward_hooks for layer in encoder.layers)


def test_duplicate_and_partial_attachment_are_safe():
    model = make_model()
    with pytest.raises(ValueError, match="already attached"):
        ResidualAdapters(model.encoder, CONFIG)
    assert all(len(layer._forward_hooks) == 1 for layer in model.encoder.layers)
    encoder = make_model(None).encoder
    with patch.object(encoder.layers[1], "register_forward_hook", side_effect=RuntimeError("failure")):
        with pytest.raises(RuntimeError, match="failure"):
            ResidualAdapters(encoder, CONFIG)
    assert all(not layer._forward_hooks for layer in encoder.layers)


def test_unsupported_encoder_and_ablation_types():
    model = make_model(None)
    model.encoder.config.model_type = "bert"
    with pytest.raises(ValueError, match="ModernBERT"):
        ResidualAdapters(model.encoder, CONFIG)
    with pytest.raises(ValueError, match="no residual"):
        model.set_residual_enabled(False)
    with pytest.raises(TypeError, match="bool"):
        make_model().set_residual_enabled(0)


def test_no_init_and_strict_checkpoint_roundtrip(tmp_path):
    original = make_model().eval()
    nonzero_branches(original)
    cfg = bundle(tmp_path, original)
    rng = torch.get_rng_state().clone()
    restored = build_model(cfg, encoder_dir=str(tmp_path / "encoder"), pretrained=False).eval()
    assert torch.equal(rng, torch.get_rng_state()), "checkpoint construction consumed RNG"
    restored.load_state_dict(original.state_dict(), strict=True)
    with torch.no_grad():
        for a, b in zip(original(*inputs()), restored(*inputs())):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    stock = make_model(None)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        stock.load_state_dict(original.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(stock.state_dict(), strict=True)


def test_sdk_load_batch_and_backend_guards(tmp_path):
    model = make_model().eval()
    nonzero_branches(model)
    bundle(tmp_path, model)
    agent = Agent(str(tmp_path), device="cpu")
    questions = {"answer": {"type": "choice", "instructions": "Select", "criteria": ["yes", "no"]}}
    states = ["item yes", "item no"]
    individual = [agent.predict(state, questions) for state in states]
    batched = agent.predict_batch(states, questions)
    for one, many in zip(individual, batched):
        assert one["answers"]["answer"]["probabilities"] == pytest.approx(
            many["answers"]["answer"]["probabilities"], abs=1e-5)
    with pytest.raises(ValueError, match="residual_adapters"):
        Agent(str(tmp_path), device="cpu", compile=True)
    with pytest.raises(ValueError, match="residual_adapters"):
        compile_model(agent.model, backend="eager")
    with pytest.raises(ValueError, match="residual_adapters"):
        agent.accelerate(strict=True)
    forward = agent.model.forward
    with pytest.warns(RuntimeWarning, match="residual_adapters"):
        assert agent.accelerate(strict=False) is False
    assert agent.model.forward == forward


def test_export_and_onnx_loader_refuse_before_execution(tmp_path):
    from scripts.export_onnx import export_to_onnx
    from laya.onnx_agent import ONNXAgent

    bundle(tmp_path)
    with patch("torch.onnx.export") as export:
        with pytest.raises(ValueError, match="residual_adapters"):
            export_to_onnx(str(tmp_path), str(tmp_path / "model.onnx"))
        export.assert_not_called()
    # The loader must reject the config before it ever asks the optional runtime for a session.
    with patch.dict(sys.modules, {"onnxruntime": types.ModuleType("onnxruntime")}):
        with pytest.raises(ValueError, match="residual_adapters"):
            ONNXAgent(str(tmp_path))


def test_sdk_refuses_incomplete_or_unconfigured_adapter_weights(tmp_path):
    cfg = bundle(tmp_path)
    cfg.pop("residual_adapters")
    (tmp_path / "rl_agent_config.json").write_text(json.dumps(cfg))
    with pytest.raises(RuntimeError, match="Unexpected key"):
        Agent(str(tmp_path), device="cpu")
    cfg["residual_adapters"] = CONFIG
    (tmp_path / "rl_agent_config.json").write_text(json.dumps(cfg))
    save_file(make_model(None).state_dict(), tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="incomplete"):
        Agent(str(tmp_path), device="cpu")


def test_split_export_refuses_residual_bundle(tmp_path):
    bundle(tmp_path)
    path = Path(__file__).resolve().parents[1] / "laya-ts/scripts/export_onnx.py"
    spec = importlib.util.spec_from_file_location("split_export_residual_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with patch("torch.onnx.export") as export:
        with pytest.raises(ValueError, match="residual_adapters"):
            module.main(["--model-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")])
        export.assert_not_called()


def test_direct_fast_constructor_refuses_before_kernel_use():
    path = Path(__file__).resolve().parents[1] / "laya/fast.py"
    spec = importlib.util.spec_from_file_location("laya._fast_residual_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"laya.tl_kernels": types.ModuleType("laya.tl_kernels")}):
        spec.loader.exec_module(module)
        with pytest.raises(ValueError, match="residual_adapters"):
            module.FastLaya(make_model())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_device_dtype_and_keyword_layer_inputs(dtype):
    model = make_model().to(dtype=dtype).eval()
    nonzero_branches(model)
    with torch.no_grad():
        logits, actions = model(*inputs()) if dtype == torch.float32 else _autocast_forward(model)
    assert torch.isfinite(logits).all() and torch.isfinite(actions).all()
    branch = model.residual_adapters.branches["0"]
    hidden = torch.randn(2, 4, 16, dtype=dtype)
    output = (torch.randn_like(hidden), "metadata")
    actual = branch.apply_to_output(None, (), {"hidden_states": hidden}, output)
    torch.testing.assert_close(actual[0], output[0] + branch(hidden))
    assert actual[1] == "metadata"


def _autocast_forward(model):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        return model(*inputs())

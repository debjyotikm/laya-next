"""Opt-in whole-quantity representations from explicit digit/place metadata.

This module does not attach itself to an Agent, discover numbers, bind fields,
convert units, or change the tokenizer. Callers own model integration and training.
"""
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .numeric import AlignedDigit


__all__ = ["QuantityBatch", "prepare_quantities", "QuantityEncoder"]


@dataclass(frozen=True)
class QuantityBatch:
    """Immutable aligned digits on a padded, batch-first token grid.

    Each row is a tuple of ``AlignedDigit`` objects. Quantity identifiers are
    local to each row; identifiers need not be consecutive. A quantity has one
    sign and at most one digit at each decimal power. Empty rows are allowed.
    ``sequence_length`` includes padding; callers must exclude padding from the
    digit metadata. Tuples can be captured safely by activation-checkpoint calls.
    """

    rows: tuple[tuple[AlignedDigit, ...], ...]
    sequence_length: int

    def __post_init__(self):
        if type(self.sequence_length) is not int or self.sequence_length < 1:
            raise ValueError("sequence_length must be a positive integer")
        if not isinstance(self.rows, tuple) or not self.rows:
            raise ValueError("rows must be a nonempty tuple of tuples")
        for row in self.rows:
            if not isinstance(row, tuple):
                raise ValueError("rows must be immutable tuples")
            quantities = {}
            for slot in row:
                if not isinstance(slot, AlignedDigit):
                    raise ValueError("rows must contain AlignedDigit values")
                if (any(type(value) is not int for value in
                        (slot.quantity, slot.token, slot.digit, slot.power)) or
                        type(slot.negative) is not bool or slot.quantity < 0 or
                        not 0 <= slot.token < self.sequence_length or not 0 <= slot.digit <= 9):
                    raise ValueError("invalid aligned digit or token-grid position")
                sign, powers = quantities.setdefault(slot.quantity, (slot.negative, set()))
                if sign != slot.negative or slot.power in powers:
                    raise ValueError("quantity has inconsistent signs or duplicate decimal powers")
                powers.add(slot.power)

    @property
    def shape(self):
        """Return the batch size and padded token length."""
        return len(self.rows), self.sequence_length


def prepare_quantities(rows: Sequence[Sequence[AlignedDigit]], sequence_length: int) -> QuantityBatch:
    """Validate and freeze caller-aligned digits without changing token positions.

    Use ``laya.numeric.align_slots`` with complete literal spans and the original
    tokenizer offsets to construct each row. No truncation or span inference is
    performed. Mutable input sequences are copied into tuples.
    """
    if (not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or
            any(not isinstance(row, Sequence) or isinstance(row, (str, bytes)) for row in rows)):
        raise ValueError("rows must be sequences of aligned-digit sequences")
    return QuantityBatch(tuple(tuple(row) for row in rows), sequence_length)


class QuantityEncoder(nn.Module):
    """Learn digit/place/sign interactions before pooling each whole quantity.

    Broadcast each quantity vector once to each token containing its digits.
    Tokens shared by multiple quantities receive their mean; unassigned tokens
    receive zero. The side vector is independent of how a quantity's digits are
    split across tokens, but this does not make the full text model invariant.

    ``rank`` is the output width, not a modification to the encoder's attention.
    Learned power embeddings have an explicit finite range. Unseen powers within
    it are not guaranteed to generalize; out-of-range powers raise rather than
    clip. Equivalent numeric spellings need not receive equivalent vectors.
    ``zero_output=True`` preserves a caller's initial residual output but delays
    gradients into the embeddings until the output projection becomes nonzero.
    """

    def __init__(self, rank: int, width: int = 32, *, min_power: int = -32,
                 max_power: int = 32, zero_output: bool = False):
        super().__init__()
        if (any(type(value) is not int for value in (rank, width, min_power, max_power)) or
                rank < 1 or width < 1 or min_power > max_power or type(zero_output) is not bool):
            raise ValueError("invalid encoder dimensions, power range or zero_output flag")
        self.rank, self.width = rank, width
        self.min_power, self.max_power = min_power, max_power
        self.zero_output = zero_output
        self.digit = nn.Embedding(10, width)
        self.place = nn.Embedding(max_power - min_power + 1, width)
        self.sign = nn.Embedding(2, width)
        self.interact = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(),
                                      nn.Linear(width, width), nn.GELU())
        self.project = nn.Linear(width, rank, bias=False)
        if zero_output:
            nn.init.zeros_(self.project.weight)

    def get_config(self) -> dict:
        """Return JSON-compatible construction metadata, separate from weights."""
        return dict(version=1, rank=self.rank, width=self.width, min_power=self.min_power,
                    max_power=self.max_power, zero_output=self.zero_output)

    @classmethod
    def from_config(cls, config: dict):
        """Construct from a versioned configuration; load weights separately."""
        keys = {"version", "rank", "width", "min_power", "max_power", "zero_output"}
        if (not isinstance(config, dict) or set(config) != keys or
                type(config["version"]) is not int or config["version"] != 1):
            raise ValueError("invalid quantity encoder configuration or version")
        return cls(**{key: value for key, value in config.items() if key != "version"})

    def forward(self, batch: QuantityBatch) -> torch.Tensor:
        """Return floating features of shape ``(batch, sequence, rank)``.

        The output follows the module's device and stored dtype. Under autocast,
        accumulation is converted to that dtype before scattering. Caller-owned
        metadata is not cached or mutated, including during checkpoint replay.
        """
        if not isinstance(batch, QuantityBatch):
            raise TypeError("expected a QuantityBatch")
        digits, powers, signs, owners, destinations, sources = [], [], [], [], [], []
        quantity_count = 0
        for row_index, row in enumerate(batch.rows):
            groups = {}
            for slot in row:
                if not self.min_power <= slot.power <= self.max_power:
                    raise ValueError("decimal power outside configured embedding range")
                groups.setdefault(slot.quantity, []).append(slot)
            for slots in groups.values():
                digits.extend(slot.digit for slot in slots)
                powers.extend(slot.power - self.min_power for slot in slots)
                signs.extend(int(slot.negative) for slot in slots)
                owners.extend([quantity_count] * len(slots))
                for token in sorted({slot.token for slot in slots}):
                    destinations.append(row_index * batch.sequence_length + token)
                    sources.append(quantity_count)
                quantity_count += 1
        weight = self.project.weight
        result = weight.new_zeros(len(batch.rows) * batch.sequence_length, self.rank)
        if not quantity_count:
            return result.reshape(*batch.shape, self.rank)

        def indices(values):
            return torch.tensor(values, dtype=torch.long, device=weight.device)

        owner = indices(owners)
        features = self.interact(torch.cat((self.digit(indices(digits)),
                                           self.place(indices(powers)), self.sign(indices(signs))), dim=-1))
        pooled = features.new_zeros(quantity_count, features.shape[-1]).index_add(0, owner, features)
        counts = torch.bincount(owner, minlength=quantity_count).to(pooled.dtype)
        projected = self.project(pooled / counts[:, None]).to(result.dtype)
        destination = indices(destinations)
        result = result.index_add(0, destination, projected[indices(sources)])
        counts = torch.bincount(destination, minlength=result.shape[0]).clamp_min(1).to(result.dtype)
        return (result / counts[:, None]).reshape(*batch.shape, self.rank)

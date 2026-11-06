"""Encode sequences using their observation times with Multi-Time Attention (mTAND).

The attention layer is adapted from FuseMoE's ``src/core/module.py``, based on
Shukla and Marlin's "Multi-Time Attention Networks for Irregularly Sampled
Time Series" (ICLR 2021, https://github.com/reml-lab/mTAN).

``mTANDEncoder`` maps observations to evenly spaced reference times.
``MTANDSequenceEncoder`` handles padded batches and returns either a pooled
vector or a flattened reference sequence. Both are used by the models in
``src/models/fusemoe.py`` and ``src/models/mtand_m3care.py``. Callers scale
observation times to match the reference range of ``[0, 1]``.

Run a small example from the repository root::

    python - <<'PY'
    import torch
    from src.encoders.mtand import MTANDSequenceEncoder

    encoder = MTANDSequenceEncoder(input_dim=3, d_model=8, tt_max=4).eval()
    feats = torch.ones(1, 2, 3)
    times = torch.tensor([[0.0, 1.0]])
    lengths = torch.tensor([2])
    with torch.no_grad():
        print(encoder(feats, times, lengths).shape)  # torch.Size([1, 8])
    PY
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class multiTimeAttention(nn.Module):
    """Weight observed features by how closely their times match each query time.

    Queries and keys represent times. Values contain the observed features.
    The same time scores are applied to each feature, with masks excluding
    missing observations before the weighted values are combined.

    Args:
        input_dim: Width of the values, including any appended mask features.
        nhidden: Output feature width after the final linear layer.
        embed_time: Width of the time representations used as queries and keys.
        num_heads: Number of attention heads. Must divide ``embed_time`` evenly.
    """

    def __init__(self, input_dim, nhidden=16, embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList(
            [
                nn.Linear(embed_time, embed_time), nn.Linear(embed_time, embed_time),
                nn.Linear(input_dim * num_heads, nhidden),
            ]
        )

    def attention(self, query, key, value, mask=None, dropout=None):
        """Return weighted values and attention weights, applying masks and dropout."""
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(-1)
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -10000)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.sum(p_attn * value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None, dropout=0.1):
        """Combine attention heads into features shaped ``[batch, queries, nhidden]``."""
        batch, _, dim = value.size()
        if mask is not None:
            # Apply the same mask to every head
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [
            l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key))
        ]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        return self.linears[-1](x)


class mTANDEncoder(nn.Module):
    """Map a sequence of observations to a fixed number of reference times.

    Represent each time with a linear component and learned sine components,
    then apply :class:`multiTimeAttention` at ``tt_max`` reference times.

    Args:
        input_dim: Number of observed features per time step.
        d_model: Output feature width.
        embed_time: Width of each time representation. Must be divisible by
            ``num_heads``.
        num_heads: Number of attention heads.
        tt_max: Number of reference times and output time steps.
        concat_mask: Append the observation mask to the values when True,
            doubling the attention value width.
        learn_query: Allow the reference times to be learned when True.
            Otherwise, keep them evenly spaced in ``[0, 1]``.
        dropout: Attention dropout probability.

    Forward inputs:
        values: Observed features shaped ``[batch, seq_len, input_dim]``.
        times: Observation times shaped ``[batch, seq_len]``. The caller should
            scale these to ``[0, 1]`` to match the initial reference times.
        mask: Optional observation mask shaped ``[batch, seq_len, input_dim]``,
            with 1 for observed features and 0 for missing features. If omitted,
            all values are treated as observed.

    Returns:
        Encoded features shaped ``[batch, tt_max, d_model]``.
    """

    def __init__(
        self, input_dim: int, d_model: int = 128, embed_time: int = 128,
        num_heads: int = 8, tt_max: int = 128, concat_mask: bool = True,
        learn_query: bool = False, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_time % num_heads != 0:
            raise ValueError(
                f"embed_time ({embed_time}) must be divisible by num_heads ({num_heads})."
            )
        self.input_dim = input_dim
        self.d_model = d_model
        self.embed_time = embed_time
        self.concat_mask = concat_mask
        self.dropout = dropout

        value_dim = input_dim * 2 if concat_mask else input_dim
        self.time_attn = multiTimeAttention(
            input_dim=value_dim,
            nhidden=d_model,
            embed_time=embed_time,
            num_heads=num_heads,
        )

        # Represent times with a linear component and learned sine components
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

        # Start with evenly spaced reference times in [0, 1]
        ref = torch.linspace(0.0, 1.0, tt_max)
        if learn_query:
            self.time_query = nn.Parameter(ref)
        else:
            self.register_buffer("time_query", ref)

    def learn_time_embedding(self, tt: torch.Tensor) -> torch.Tensor:
        """Expand times from ``[..., L]`` to ``[..., L, embed_time]`` features."""
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(
        self, values: torch.Tensor, times: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode values at the reference times, masking missing observations."""
        time_key = self.learn_time_embedding(times)
        # Share reference times across the batch
        time_query = self.learn_time_embedding(self.time_query.unsqueeze(0))

        if self.concat_mask:
            if mask is None:
                mask = torch.ones_like(values)
            att_values = torch.cat([values, mask], dim=-1)
            att_mask = torch.cat([mask, mask], dim=-1)
        else:
            att_values = values
            att_mask = mask

        dropout = self.dropout if self.training else None
        return self.time_attn(time_query, time_key, att_values, att_mask, dropout)


class MTANDSequenceEncoder(nn.Module):
    """Encode padded feature sequences using their observation times.

    Use ``concat_mask=False`` for ECG and CXR embeddings. For time series with
    missing features, use ``concat_mask=True`` and pass ``obs_mask`` to
    ``forward``. Record lengths identify padding in both cases.

    Mean and max pooling return one vector per sample. With ``pool="none"``,
    normalise and flatten the reference sequence in time order. Pooled modes
    apply normalisation after pooling.

    Args:
        input_dim: Number of features per record.
        d_model: Encoded feature width at each reference time.
        embed_time: Width of each time representation. Must be divisible by
            ``num_heads``.
        num_heads: Number of attention heads.
        tt_max: Number of reference times.
        pool: ``"mean"`` or ``"max"`` to pool over reference times,
            or ``"none"`` to flatten the sequence.
        concat_mask: Append the observation mask to the attention values when
            True. This doubles the value width without changing the output width.
        dropout: Dropout probability for attention and output features.

    Attributes:
        output_dim: Output width, ``d_model`` when pooled or
            ``tt_max * d_model`` when flattened.

    Forward inputs:
        feats: Padded features shaped ``[B, L, input_dim]``.
        time: Observation times shaped ``[B, L]``, scaled by the caller to
            match the reference range of ``[0, 1]``.
        lengths: Number of real records per sample, shaped ``[B]``.
        obs_mask: Optional mask shaped ``[B, L, input_dim]``, with 1 for observed
            features and 0 for missing features. Combined with the padding mask.

    Returns:
        Features shaped ``[B, d_model]`` when pooled, or
        ``[B, tt_max * d_model]`` when flattened.
    """

    def __init__(
        self, input_dim: int, d_model: int = 128, embed_time: int = 128,
        num_heads: int = 8, tt_max: int = 128, pool: str = "mean",
        dropout: float = 0.0, concat_mask: bool = False,
    ) -> None:
        super().__init__()
        if pool not in ("mean", "max", "none"):
            raise ValueError(f"pool must be 'mean', 'max', or 'none', got {pool!r}.")
        self.pool = pool
        self.d_model = d_model
        self.tt_max = tt_max
        self.concat_mask = concat_mask
        self.encoder = mTANDEncoder(
            input_dim=input_dim,
            d_model=d_model,
            embed_time=embed_time,
            num_heads=num_heads,
            tt_max=tt_max,
            concat_mask=concat_mask,
            learn_query=False,
            dropout=dropout,
        )
        # Normalise over features in both pooled and flattened outputs
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    @property
    def output_dim(self) -> int:
        """Return the feature width of the pooled or flattened output."""
        if self.pool == "none":
            return self.tt_max * self.d_model
        return self.d_model

    @staticmethod
    def _padding_mask(time: torch.Tensor, lengths: torch.Tensor, input_dim: int) -> torch.Tensor:
        """Mark real records with 1 and padding with 0 in ``[B, L, input_dim]``."""
        max_len = time.shape[1]
        idx = torch.arange(max_len, device=time.device).unsqueeze(0)
        valid = idx < lengths.to(device=time.device).unsqueeze(1)
        return valid.unsqueeze(-1).expand(-1, -1, input_dim).to(dtype=time.dtype)

    def forward(
        self, feats: torch.Tensor, time: torch.Tensor, lengths: torch.Tensor,
        obs_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mask padded and missing features, then pool or flatten the encoded sequence."""
        if feats.ndim != 3:
            raise ValueError("feats must have shape (B, L, F).")
        mask = self._padding_mask(time, lengths, feats.shape[-1])
        if obs_mask is not None:
            # Keep features marked as observed only within real records
            if obs_mask.shape != feats.shape:
                raise ValueError("obs_mask must have the same shape as feats (B, L, F).")
            mask = mask * obs_mask.to(dtype=mask.dtype)
        ref = self.encoder(feats, time, mask)
        if self.pool == "none":
            # Normalise each reference step before flattening in time order
            ref = self.drop(self.act(self.norm(ref)))
            return ref.reshape(ref.shape[0], -1)
        # Pool over reference times before normalising the result
        pooled = ref.mean(dim=1) if self.pool == "mean" else ref.max(dim=1).values
        return self.drop(self.act(self.norm(pooled)))

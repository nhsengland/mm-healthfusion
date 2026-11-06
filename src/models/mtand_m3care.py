from typing import Any, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
from torch.nn import Parameter
from collections.abc import Mapping
from src.models.base import BaseMultimodalModel
from src.data.collate import sequence_available, tabular_available
from src.encoders.mtand import MTANDSequenceEncoder

MODALITIES = ("tabular", "ecg", "cxr", "ts")

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class MM_transformer_encoder(nn.Module):
    def __init__(self, input_dim, d_model, MHD_num_head, d_ff, output_dim, dropout=0.5):
        super(MM_transformer_encoder, self).__init__()

        # hyperparameters
        self.input_dim = input_dim  
        self.d_model = d_model
        self.MHD_num_head = MHD_num_head
        self.d_ff = d_ff
        self.dropout_prob = dropout

        # layers
        self.embed = nn.Linear(self.input_dim, self.d_model)

        self.PositionalEncoding = PositionalEncoding(self.d_model, dropout = 0, max_len = 5000)
       
        self.MultiHeadedAttention = MultiHeadedAttention(self.MHD_num_head, self.d_model)
        self.SublayerConnection = SublayerConnection(self.d_model, dropout = self.dropout_prob)

        self.PositionwiseFeedForward = PositionwiseFeedForward(self.d_model, self.d_ff, dropout=0.1)


    def forward(self, input, mask, return_attn: bool = False):
        # input shape [batch_size, timestep, feature_dim]
        if input.ndim != 3:
            raise ValueError("MM_transformer_encoder expects input with shape (B, T, F).")
        
        feature_dim = input.size(2)
        
        if self.input_dim is not None and feature_dim != self.input_dim:
            raise ValueError(
                f"Expected feature_dim={self.input_dim}, got {feature_dim}."
            )
        assert self.d_model % self.MHD_num_head == 0

     
        input = self.embed(input)
        input = self.PositionalEncoding(input) # (B, T, d_model)

        contexts = self.SublayerConnection(input, lambda input: self.MultiHeadedAttention(input, input, input, mask)) # (B, T, d_model)
        attn = self.MultiHeadedAttention.attn  # (B, heads, T, T)

        contexts = self.SublayerConnection(contexts, lambda contexts: self.PositionwiseFeedForward(contexts)) # (B, T, d_model)
        
        if return_attn:
            return contexts, attn
        return contexts # (B, T, H)


def euclidean_dist(x, y):
    b = x.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(b, b)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(b, b).t()
    dist = xx+yy-2*torch.mm(x, y.t())
    return dist 

def gaussian_kernel(source, kernel_mul=2.0, kernel_num=1, fix_sigma=None):
    """
    x: (B, H)
    returns: (B, B) similarity matrix
    """
    if source.ndim != 2:
        raise ValueError("gaussian_kernel expects source with shape (B, H).")
    n = int(source.size(0))
    L2_distance = euclidean_dist(source, source)

    if fix_sigma is not None:
        bandwidth = torch.as_tensor(fix_sigma, device=source.device, dtype=source.dtype)
    else:
        denom = float(max(n * (n - 1), 1))
        bandwidth = L2_distance.detach().sum() / denom

    if float(bandwidth.item()) < 1e-3:
        bandwidth = bandwidth.new_tensor(1.0)
    bandwidth /= kernel_mul ** (kernel_num//2)
    bandwidth_list = [bandwidth*(kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance/bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)/len(kernel_val)


def to_binary_mask(mask):
    """
    Convert input mask to shape (B, 1) and type float/int-friendly.
    """
    if mask is None:
        return None
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    if mask.ndim == 1:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 2 or mask.shape[1] != 1:
        raise ValueError("Mask must have shape (B,) or (B, 1).")
    return mask.float()


def pairwise_mask(mask_1):
    """
    mask_1: (B, 1)
    returns: (B, B)
    """
    if mask_1.ndim == 1:
        mask_1 = mask_1.unsqueeze(1)
    elif mask_1.ndim != 2 or mask_1.shape[1] != 1:
        raise ValueError("mask_1 must have shape (B,) or (B, 1).")
    return mask_1 * mask_1.permute(1, 0)
        


def cosine_similarity_matrix(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return the pairwise cosine similarity matrix for rows of x."""
    if x.ndim != 2:
        raise ValueError("cosine_similarity_matrix expects x with shape (B, H).")
    x = F.normalize(x, p=2, dim=-1, eps=eps)
    return x @ x.t()

def check_modality_keys(
    expected_modalities: Sequence[str],
    modality_embeddings: Mapping[str, torch.Tensor],
    modality_masks: Mapping[str, torch.Tensor],
) -> None:
    expected = set(expected_modalities)
    emb_keys = set(modality_embeddings.keys())
    mask_keys = set(modality_masks.keys())

    if emb_keys != expected:
        raise KeyError(
            f"Embedding keys mismatch.\n"
            f"Expected: {sorted(expected)}\n"
            f"Got: {sorted(emb_keys)}"
        )

    if mask_keys != expected:
        raise KeyError(
            f"Mask keys mismatch.\n"
            f"Expected: {sorted(expected)}\n"
            f"Got: {sorted(mask_keys)}"
        )

def _resolve_modality_input_sizes(
    modality_names: Sequence[str],
    modality_input_sizes: int | Sequence[int] | Mapping[str, int] | None,
    default_size: int | None = None,
) -> dict[str, int]:
    """Normalize modality-specific widths into an ordered name -> size mapping."""
    names = list(modality_names)

    if modality_input_sizes is None:
        if default_size is None:
            raise ValueError(
                "modality_input_sizes must be provided when modality widths vary."
            )
        return {name: int(default_size) for name in names}

    if isinstance(modality_input_sizes, Mapping):
        missing = [name for name in names if name not in modality_input_sizes]
        extra = [name for name in modality_input_sizes.keys() if name not in names]
        if missing or extra:
            raise KeyError(
                "modality_input_sizes keys must match modality_names exactly. "
                f"Missing: {missing or 'none'}; extra: {extra or 'none'}"
            )
        return {name: int(modality_input_sizes[name]) for name in names}

    if isinstance(modality_input_sizes, int):
        return {name: int(modality_input_sizes) for name in names}

    if isinstance(modality_input_sizes, Sequence) and not isinstance(
        modality_input_sizes, (str, bytes)
    ):
        values = list(modality_input_sizes)
        if len(values) != len(names):
            raise ValueError(
                "When modality_input_sizes is a sequence, it must have one entry "
                f"per modality. Expected {len(names)}, got {len(values)}."
            )
        return {name: int(size) for name, size in zip(names, values)}

    raise TypeError(
        "modality_input_sizes must be a mapping, sequence, integer, or None."
    )


def check_subset_modality_keys(
    expected_modalities: Sequence[str],
    modality_embeddings: Mapping[str, torch.Tensor],
    modality_masks: Mapping[str, torch.Tensor],
) -> None:
    """Validate that the provided dictionaries contain exactly the requested subset."""
    expected = list(expected_modalities)
    emb_keys = list(modality_embeddings.keys())
    mask_keys = list(modality_masks.keys())

    if set(emb_keys) != set(expected):
        raise KeyError(
            "Embedding keys mismatch.\n"
            f"Expected: {sorted(expected)}\n"
            f"Got: {sorted(emb_keys)}"
        )

    if set(mask_keys) != set(expected):
        raise KeyError(
            "Mask keys mismatch.\n"
            f"Expected: {sorted(expected)}\n"
            f"Got: {sorted(mask_keys)}"
        )

    
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features).float())
        if bias:
            self.bias = Parameter(torch.Tensor(out_features).float())
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        std = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, adj, x):
        y = torch.mm(x.float(), self.weight.float())
        output = torch.mm(adj.float(), y.float())
        if self.bias is not None:
            return output + self.bias.float()
        else:
            return output


class GCN_imputation(nn.Module):
    """
    Modality-specific GCN imputation without forcing a shared width.

    Expects:
      - modality_embeddings: dict[str, Tensor] with each Tensor shaped (B, H)
      - modality_masks:      dict[str, Tensor] with each Tensor shaped (B, 1)

    Returns:
      - dict of refined embeddings keyed by modality name
      - stability_loss scalar tensor
      - imputation attentions per modality
    """
    def __init__(
        self,
        # hidden_dim: int,
        modality_names: tuple[str, ...] | list[str], 
        modality_input_sizes: int | Sequence[int] | Mapping[str, int] | None,
        graph_construction_loss_weight: float = 0.01,
        stability_loss_type: str = "norm",
        gcn_refine_mode: str = "adaptive",
        modalities_to_impute: tuple[str, ...] | list[str] = None,
    ):
        super().__init__()

        self.modality_names = list(modality_names)
        self.modal_num = len(self.modality_names)

        self.modality_input_sizes = _resolve_modality_input_sizes(
            self.modality_names,
            modality_input_sizes,
        )

        self.graph_construction_loss_weight = graph_construction_loss_weight
        self.stability_loss_type = stability_loss_type.lower().strip()
        self.gcn_refine_mode = gcn_refine_mode.lower().strip()
        if self.gcn_refine_mode not in {"adaptive", "imputed_only", "imputed_for_missingness"}:
            raise ValueError(
                "gcn_refine_mode must be one of: 'adaptive', 'imputed_only', 'imputed_for_missingness'."
            )
        
        if modalities_to_impute is None:
            # Default: impute all modalities
            self.modalities_to_impute = list(self.modality_names)
            self.modal_impute_num = len(self.modalities_to_impute)
            print(
                "Since modalities_to_impute is not specified, "
                f"default to imputing all modalities: {self.modalities_to_impute}"
            )

        elif len(modalities_to_impute) == 0:
            raise ValueError(
                "modalities_to_impute cannot be an empty list. "
                "Either set it to None to impute all modalities, "
                "or provide at least one modality name."
            )

        elif set(modalities_to_impute).issubset(self.modality_names):
            self.modalities_to_impute = [
                m for m in self.modality_names if m in modalities_to_impute
            ]
            self.modal_impute_num = len(self.modalities_to_impute)
            print(f"The modalities that will be imputed: {self.modalities_to_impute}")

        else:
            invalid = sorted(set(modalities_to_impute).difference(self.modality_names))
            raise ValueError(
                f"Unknown modalities in modalities_to_impute: {invalid}. "
                f"Available modalities are: {self.modality_names}."
            )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.simiProj = nn.ModuleDict()
        self.bn = nn.ModuleDict()
        self.eps_params = nn.ParameterDict()
        for name in self.modality_names:
            width = self.modality_input_sizes[name]
            # Similarity projection heads (one per modality)
            self.simiProj[name] = nn.Sequential(
                nn.Linear(width, width, bias=True),
                nn.ReLU(),
                nn.Linear(width, width, bias=True),
                nn.ReLU(),
                nn.Linear(width, width, bias=True),
            )
            self.bn[name] = nn.BatchNorm1d(width)
            # Learnable mixing parameters
            self.eps_params[name] = nn.Parameter(torch.ones(size=(1,)))
        
        # Similarity projection heads (one per modality)
        self.GCN = nn.ModuleDict()
        self.GCN2 = nn.ModuleDict()
        for name in self.modalities_to_impute:
            width = self.modality_input_sizes[name]
            self.GCN[name] = GraphConvolution(width, width, bias=True)
            self.GCN2[name] = GraphConvolution(width, width, bias=True)

        self.threshold = nn.Parameter(torch.zeros(size=(1,)))

        # Per-modality gating for refined vs original embeddings
        if self.gcn_refine_mode == "adaptive":
            # Weights only for modalities selected for imputation
            self.weight1 = nn.ModuleDict()
            self.weight2 = nn.ModuleDict()
            for name in self.modalities_to_impute:
                width = self.modality_input_sizes[name]
                self.weight1[name] = nn.Linear(width, 1)
                self.weight2[name] = nn.Linear(width, 1)
        else:
            # Since no adaptive fusion will be performed, the weights are set to None as placeholder
            self.weight1 = None
            self.weight2 = None
    
    def _compute_stability_loss(self, x, proj_x, proj_x_for_graph):
        """
        Compute the stability regulariser for one modality.

        Supported choices:
          - norm: original norm-based loss from M3Care
          - cosine: cosine-similarity matrix preservation loss
          - similarity: Gaussian-kernel similarity matrix preservation loss
        """
        loss_type = self.stability_loss_type
        if loss_type == "norm":
            loss = torch.abs(torch.norm(proj_x) - torch.norm(x))
        elif loss_type == "cosine":
            x_cos = cosine_similarity_matrix(x)
            p_cos = cosine_similarity_matrix(proj_x_for_graph)
            loss = F.mse_loss(p_cos, x_cos)
        elif loss_type in {"similarity", "matrix", "similarity_matrix"}:
            x_sim = gaussian_kernel(x, kernel_mul=2.0, kernel_num=3)
            p_sim = gaussian_kernel(proj_x_for_graph, kernel_mul=2.0, kernel_num=3)
            loss = F.mse_loss(p_sim, x_sim)
        else:
            raise ValueError(
                f"Unknown stability_loss_type='{self.stability_loss_type}'. "
                "Use one of: 'norm', 'cosine', 'similarity' (aliases: 'matrix', 'similarity_matrix')."
            )
        return loss * self.graph_construction_loss_weight
        
    
    def _build_similarity(self, name, x):
        """
        x: (B, H)
        Returns:
          sim: batch similarity matrix (B, B)
          stability_loss: scalar
        """
        proj = self.simiProj[name]
        bn = self.bn[name]
        eps_param = self.eps_params[name]
        proj_x = proj(x)
        proj_x_for_graph = bn(self.relu(proj_x))
        x_for_graph = bn(x)

        sim1 = gaussian_kernel(proj_x_for_graph, kernel_mul=2.0, kernel_num=3)
        sim2 = gaussian_kernel(x_for_graph, kernel_mul=2.0, kernel_num=3)

        gate = self.sigmoid(eps_param)
        sim = ((1 - gate) * sim1 + gate) * sim2
        stability_loss = self._compute_stability_loss(x, proj_x, proj_x_for_graph)
        return sim, stability_loss

    
    def _refine_with_gcn(self, name, sim_score, x, mask):
        """
        sim_score: (B, B)
        x: (B, H)
        """
        x0 = self.relu(self.GCN[name](sim_score, x))
        x0 = self.relu(self.GCN2[name](sim_score, x0))
        
        # Force missing-modality samples to use x0 only
        mask = mask.to(device=x.device, dtype=x.dtype)

        if self.gcn_refine_mode == "adaptive":
            # M3care:
            # - present samples: adaptive mix of x0 and x
            # - missing samples: x0 only
            w1 = torch.sigmoid(self.weight1[name](x0))
            w2 = torch.sigmoid(self.weight2[name](x))
            w1 = w1 / (w1 + w2 + 1e-8)
            w2 = 1.0 - w1

            w1 = w1 * mask + (1.0 - mask)
            w2 = w2 * mask
            x_final = w1 * x0 + w2 * x

        elif self.gcn_refine_mode == "imputed_only":
            # Always use the GCN-imputed embedding
            w1 = torch.ones_like(mask)
            w2 = torch.zeros_like(mask)
            x_final = x0

        elif self.gcn_refine_mode == "imputed_for_missingness":
            # Present samples use the original embedding; missing samples use the imputed embedding.
            # This is a hard routing rule, so no learned/adaptive weights are used here.
            w1 = 1.0 - mask
            w2 = mask
            x_final = w1 * x0 + w2 * x

        else:
            raise ValueError(
                f"Unknown gcn_refine_mode='{self.gcn_refine_mode}'. "
                "Use one of: 'adaptive', 'imputed_only', 'imputed_for_missingness'."
            )

        return x0, x_final, [w1, w2]


    def forward(
        self, 
        modality_embeddings: Mapping[str, torch.Tensor], 
        modality_masks: Mapping[str, torch.Tensor]
        ): 
        
        check_modality_keys(self.modality_names, modality_embeddings, modality_masks)

        sim_list = []
        mask_list = []
        stability_losses = []

        for name in self.modality_names:
            x = modality_embeddings[name]
            mask = modality_masks[name]
            expected = self.modality_input_sizes[name]
            if x.ndim != 2:
                raise ValueError(f"Modality '{name}' embeddings must have shape (B, H).")
            if x.size(-1) != expected:
                raise ValueError(f"Modality '{name}' expected width {expected}, got {x.size(-1)}.")
            
            # 1) Build pairwise missing-modality masks: (B, B)
            mask2 = pairwise_mask(mask)

            # 2) Build pairwise similarity matrices for each modality
            sim, stability_loss = self._build_similarity(name, x)
            sim = sim * mask2 # Remove graph edges involving missingness
            sim_list.append(sim)
            mask_list.append(mask2)
            stability_losses.append(stability_loss)
        
        # 3) Compute combined similarity score for inputs in the batch and the sum of stability losses
        sum_masks = torch.stack(mask_list, dim=0).sum(dim=0).clamp_min(1.0)
        similar_score = torch.stack(sim_list, dim=0).sum(dim=0) / sum_masks
        sum_stability_losses = torch.stack(stability_losses, dim=0).sum()

        # 4) Thresholding similar to original M3Care
        threshold = self.sigmoid(self.threshold)[0]
        similar_score = self.relu(similar_score - threshold)
        bin_mask = similar_score > 0
        similar_score = similar_score + bin_mask * threshold.detach()

        # 5) Graph convolution refinement + residual mixing
        refined_outputs = {}
        imputation_attentions = {}
        # Perform imputation for selected modalities
        for name in self.modality_names:
            if name in self.modalities_to_impute:
                _, x_final, attns = self._refine_with_gcn(
                    name,
                    similar_score,
                    modality_embeddings[name],
                    modality_masks[name],
                )
                refined_outputs[name] = x_final
                imputation_attentions[name] = attns
            else:
                refined_outputs[name] = modality_embeddings[name]

        return refined_outputs, sum_stability_losses, imputation_attentions



class MHSA_Fusion(nn.Module):
    """
    Fuse only the aligned temporal modalities with no width projection.

    Expects:
      - modality_embeddings: dict[str, Tensor] with each Tensor shaped (B, H)
      - modality_masks:      dict[str, Tensor] with each Tensor shaped (B, 1)

    The modality order is defined by `modality_names`, and the input dictionaries
    must use the same keys.

    Returns:
      - dict of fused embeddings keyed by modality name
    """

    def __init__(
        self,
        modality_input_sizes: int | Sequence[int] | Mapping[str, int] | None,
        # hidden_dim: int,
        modality_names: tuple[str, ...] | list[str],
        MHD_num_head: int = 4,
        dropout: float = 0.0,
        missingness_embedding: bool = False,
        mask_missingness_fusion: bool = False,
        fusion_dim: int | None = None,
    ):
        super().__init__()

        self.modality_names = list(modality_names)
        self.num_modalities = len(self.modality_names)

        self.modality_input_sizes = _resolve_modality_input_sizes(
            self.modality_names,
            modality_input_sizes,
            default_size=fusion_dim,
        )
        sizes = set(self.modality_input_sizes.values())

        if fusion_dim is None:
            if len(sizes) != 1:
                raise ValueError(
                    "MHSA_Fusion requires all fusion modalities to share the same embedding width when no fusion_dim is provided."
                )
            self.fusion_dim = int(next(iter(sizes)))
        else:
            self.fusion_dim = int(fusion_dim)
            mismatch = {name: size for name, size in self.modality_input_sizes.items() if size != self.fusion_dim}
            if mismatch:
                raise ValueError(
                    "All fusion modalities must match fusion_dim when no input projection is used. "
                    f"Got mismatches: {mismatch}."
                )
        
        self.missingness_embedding = missingness_embedding
        self.mask_missingness_fusion = mask_missingness_fusion

        self.token_type_embeddings = nn.Embedding(self.num_modalities, self.fusion_dim)
        self.token_type_embeddings.apply(init_weights)
        if self.missingness_embedding:
            self.missingness_embeddings = nn.Embedding(2, self.fusion_dim)
            self.missingness_embeddings.apply(init_weights)

        self.MM_model = MM_transformer_encoder(
            input_dim=self.fusion_dim,
            d_model=self.fusion_dim,
            MHD_num_head=MHD_num_head,
            d_ff=self.fusion_dim * MHD_num_head,
            output_dim=self.fusion_dim,
            dropout=dropout,
        )
        self.MM_model2 = MM_transformer_encoder(
            input_dim=self.fusion_dim,
            d_model=self.fusion_dim,
            MHD_num_head=1,
            d_ff=self.fusion_dim * MHD_num_head,
            output_dim=self.fusion_dim,
            dropout=dropout,
        )
        self.relu = nn.ReLU()

    def _make_token(self, x, modality_id, present_mask, batch_size, device):
        """
        x: (B, H)
        present_mask: (B, 1), 1=present, 0=missing
        """
        if self.missingness_embedding:
            presence_id = present_mask.long().clamp(0, 1).squeeze(1) # (B,)
            x = x + self.missingness_embeddings(presence_id)  # add presence signal

        x = x.unsqueeze(1) # (B, 1, H)
        modality_ids = torch.full((batch_size, 1), modality_id, dtype=torch.long, device=device)
        x = x + self.token_type_embeddings(modality_ids)
        return x

    def forward(
            self, 
            modality_embeddings: Mapping[str, torch.Tensor], 
            modality_masks: Mapping[str, torch.Tensor],
            return_attentions: bool = False,
    ) -> dict[str, Union[torch.Tensor, dict]]:
        
        check_modality_keys(self.modality_names, modality_embeddings, modality_masks)

        first_key = self.modality_names[0]
        first_embed = modality_embeddings[first_key]
        if first_embed.ndim != 2:
            raise ValueError("Modality embeddings must have shape (B, H): precomputed embeddings only.")

        batch_size = first_embed.size(0)
        device = first_embed.device

        token_list = []
        # 1) Turn each modality into one token
        for idx, name in enumerate(self.modality_names):
            x = modality_embeddings[name]
            mask = modality_masks[name]
            if x.ndim != 2:
                raise ValueError("Modality embeddings must have shape (B, H): precomputed embeddings only.")
            if mask.ndim != 2 or mask.shape[1] != 1:
                raise ValueError("Modality masks must have shape (B, 1).")
            expected = self.modality_input_sizes[name]
            if x.size(-1) != expected:
                raise ValueError(
                    f"Fusion modality '{name}' expected width {expected}, got {x.size(-1)}."
                )
            token_list.append(self._make_token(x, idx, mask, batch_size, device))
        
        # 2) Concatenate as a multi-token sequence
        z0 = torch.cat(token_list, dim=1)  # (B, M, H)

        if self.mask_missingness_fusion:
            # Use the original modality availability mask
            z0_mask = torch.cat(
                [modality_masks[name].int() for name in self.modality_names],
                dim=-1,
            ).unsqueeze(1)
        else:
            # Generate an all-one mask to ensure every modality participates in fusion
            z0_mask = torch.ones(
                batch_size,
                1,
                self.num_modalities,
                dtype=torch.int,
                device=device,
            )
        
        # 3) Two-stage transformer fusion
        z1, attn_stage1 = self.MM_model(z0, z0_mask, return_attn=True)
        z1 = self.relu(z1)

        z2, attn_stage2 = self.MM_model2(z1, z0_mask, return_attn=True)
        z2 = self.relu(z2)
        
        # 4) Pull out modality-specific fused tokens
        fused_tokens = {name: z2[:, idx, :] for idx, name in enumerate(self.modality_names)}

        # 5) Embedding concatenation
        combined_hidden = torch.cat([fused_tokens[name] for name in self.modality_names], dim=-1)

        if return_attentions:
            # attn tensors are (B, heads, Q, K)
            return {
                "embeds": combined_hidden,
                "fusion_attentions": {
                    "stage_1": attn_stage1,
                    "stage_2": attn_stage2,
                    "modalities": self.modality_names,
                }
            }

        return {
            "embeds": combined_hidden,
        }


class mTAND_M3CareModel(BaseMultimodalModel):
    """
    Multimodal fusion model for pre-extracted embeddings with mTAND time series imputation:
      - tabular:  (B, tabular_dim)
      - ecg:      (B, R, ecg_dim)
      - cxr:      (B, R, cxr_dim)
      - ts:       (B, R, ts_dim)

    This version keeps tabular outside MHSA fusion and fuse only ecg/cxr/ts.

    It:
      1) projects ecg, cxr and is to a shared hidden space
      2) builds pairwise similarity matrices within the batch for all modalities
      3) computes a similarity score / adjacency matrix
      4) applies graph convolution to refine embeddings
      5) mixes refined and original embeddings
      6) fuses ecg, cxr and ts's modality tokens with a Transformer
      7) concatenate tabular's embedding to the fused embeddings of ecg, cxr and ts
      8) predicts the final output
    """
    supports_aux_output = True

    def __init__(
            self,
            tabular_dim: int,
            ecg_dim: int,
            cxr_dim: int,
            d_model: int,
            modality_names: tuple[str, ...] | list[str],
            output_dim: int = 1,
            dropout: float = 0.0,
            MHD_num_head: int = 4,
            use_modality_mask: bool = False,
            impute: bool = True,
            missingness_embedding: bool = False,
            graph_construction_loss_weight: float = 0,
            stability_loss_type: str = "norm",
            gcn_refine_mode: str = "adaptive",
            mask_missingness_fusion: bool = False,
            modalities_to_impute: tuple[str, ...] | list[str] = None,
            use_mtand: bool = True,
            mtand_tt_max: int = 64,
            mtand_embed_time: int = 64,
            mtand_num_heads: int = 8,
            mtand_time_scale: float = 90.0,
            mtand_pool: str = "none",
            ts_modality_enabled: bool = False,
            ts_modality_summary_dim: int | None = None,
            ts_modality_summary_trailing_skip: int = 0,
            ts_modality_concat_mask: bool = True,
            ts_dim: int | None = None,
            fusion_modality_names: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        super().__init__(
            tabular_dim=tabular_dim,
            ecg_dim=ecg_dim,
            cxr_dim=cxr_dim,
            d_model=d_model,
            dropout=dropout,
        )
        # self.hidden_dim = d_model
        self.output_dim = output_dim
        self.modality_names = list(modality_names)

        if ts_modality_enabled:
            if "ts" not in self.modality_names:
                self.modality_names.append("ts")
        elif "ts" in self.modality_names:
            raise ValueError(
                "'ts' is included in modality_names, but "
                "ts_modality.enabled is false."
            )

        self.modal_num = len(self.modality_names)
        self.perform_imputation = impute
        self.graph_construction_loss_weight = graph_construction_loss_weight
        self.stability_loss_type = stability_loss_type.lower().strip()
        self.gcn_refine_mode = gcn_refine_mode.lower().strip()
        self.modalities_to_impute = modalities_to_impute

        if fusion_modality_names is None:
            self.fusion_modality_names = [name for name in self.modality_names if name != "tabular"]

        else:
            self.fusion_modality_names = list(fusion_modality_names)
            if "tabular" in self.fusion_modality_names and bool(use_mtand):
                raise ValueError("tabular must not be part of MHSA fusion when mTAND is enabled.")
            if not set(self.fusion_modality_names).issubset(self.modality_names):
                invalid = sorted(set(self.fusion_modality_names).difference(self.modality_names))
                raise ValueError(f"Unknown fusion modalities: {invalid}.")
        
        if len(self.fusion_modality_names) == 0:
            raise ValueError(
                "fusion_modality_names cannot be an empty list ([])."
            )

        self.use_fusion = len(self.fusion_modality_names) > 1

        if self.perform_imputation == False and self.modalities_to_impute:
            print(f"Imputation will not be performed, so the non-empty modalities_to_impute attribute will be ignored.")

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.use_modality_mask = use_modality_mask

        ####### Copied from fusemoe.py #######

        # Time-series modality configuration:
        # - TS disabled: time-series summaries remain within tabular features.
        # - TS enabled with mTAND: raw time-series sequences are encoded
        #   as a separate temporal modality.
        #
        # Static TS summary extraction (TS enabled without mTAND)
        # is retained below for reference but is not supported.
        self.ts_modality_enabled = bool(ts_modality_enabled)
        self.ts_modality_summary_dim: int | None = None
        self.ts_modality_summary_trailing_skip: int = 0
        self.ts_dim: int | None = None
        self.ts_modality_concat_mask = bool(ts_modality_concat_mask)
        # A separate TS modality is supported only when mTAND is enabled.
        self.ts_temporal = self.ts_modality_enabled and bool(use_mtand)
        # Static TS is retained below for reference but is not supported.
        if self.ts_modality_enabled and not use_mtand:
            raise ValueError(
                "A separate TS modality requires mtand.enabled: true. "
                "Static TS summary extraction is not supported."
            )
        # if self.ts_modality_enabled and not use_mtand:
        #     # ---- T2 static-summary path ----
        #     # The summary block is the `summary_dim` labvalues_+vitals_ columns.
        #     # In a normalised+OHE bundle it is NOT the very last block: OHE
        #     # categoricals (e.g. demographics_ethnicity_*) are appended after it
        #     # (split_deterioration export order = numeric-selected + OHE), so the
        #     # block sits `summary_trailing_skip` columns from the end. Layout:
        #     #   [ leading static | summary_dim summary | summary_trailing_skip OHE ]
        #     # The `ts` modality is the middle span; the `tabular` modality is the
        #     # leading static columns concatenated with the trailing OHE columns.
        #     skip = 0 if ts_modality_summary_trailing_skip is None else int(
        #         ts_modality_summary_trailing_skip
        #     )
        #     if ts_modality_summary_dim is None or int(ts_modality_summary_dim) <= 0:
        #         raise ValueError(
        #             "ts_modality.enabled requires a positive ts_modality.summary_dim "
        #             "(the width of the labvalues_+vitals_ summary block in "
        #             f"tab_feats); got {ts_modality_summary_dim!r}."
        #         )
        #     if skip < 0:
        #         raise ValueError(
        #             "ts_modality.summary_trailing_skip must be >= 0 (the count of "
        #             "trailing non-summary columns, e.g. OHE categoricals, that follow "
        #             f"the summary block); got {ts_modality_summary_trailing_skip!r}."
        #         )
        #     if int(ts_modality_summary_dim) + skip >= tabular_dim:
        #         raise ValueError(
        #             "ts_modality.summary_dim + summary_trailing_skip must be smaller "
        #             f"than the tabular width ({tabular_dim}) so the tabular modality "
        #             f"is non-empty; got summary_dim={ts_modality_summary_dim}, "
        #             f"summary_trailing_skip={skip}."
        #         )
        #     self.ts_modality_summary_dim = int(ts_modality_summary_dim)
        #     self.ts_modality_summary_trailing_skip = skip
        #     # Reduced tabular encoder (leading static + trailing OHE) plus a ts
        #     # encoder for the summary span. Both are flat static vectors, so the
        #     # ts modality reuses TabularEncoder. Assigning over the base-built
        #     # `tabular_encoder` keeps the change local to this model.
        #     self.tabular_encoder = TabularEncoder(
        #         input_dim=tabular_dim - self.ts_modality_summary_dim,
        #         d_model=d_model,
        #         dropout=dropout,
        #     )
        #     self.ts_encoder = TabularEncoder(
        #         input_dim=self.ts_modality_summary_dim,
        #         d_model=d_model,
        #         dropout=dropout,
        #     )
        if self.ts_temporal:
            # ---- T3 temporal path ---- (raw ts sequence via mTAND; the encoder
            # itself is built in the use_mtand block below, alongside ECG/CXR,
            # because it needs the same mtand_* settings). Here we only validate
            # that the bundle actually exposes the raw sequence.
            if ts_dim is None or int(ts_dim) <= 0:
                raise ValueError(
                    "temporal ts_modality (ts_modality.enabled + mtand.enabled) "
                    "requires a --record all bundle exposing ts_feats; feature_dims "
                    f"has no positive 'ts' width (got ts_dim={ts_dim!r})."
                )
            self.ts_dim = int(ts_dim)
        
        # -----------------------------------------------------------------
        # AMENDED FOR TIME-AWARE mTAND ECG/CXR fusion:
        # When use_mtand is True, ECG and CXR are encoded by time-aware mTAND.
        # How the mTAND reference sequence is reduced is controlled by
        # mtand_pool:
        #   "none" (FuseMoE-faithful): flatten the [B, R, d_model] reference
        #     sequence to [B, R*d_model] (R = mtand_tt_max), exactly as
        #     upstream FuseMoE flattens each mTAND-projected modality before
        #     concatenating into one MoE. Tabular stays a flat [B, d_model]
        #     vector (genuinely non-temporal here), giving UNEQUAL per-modality
        #     widths -- supported only by the joint/combination routers.
        #   "mean"/"max": pool the reference sequence back to [B, d_model], so
        #     every modality is d_model wide again. This keeps the MoE the same
        #     size as the mean-pooling baseline (a size-matched, controlled
        #     comparison isolating time-awareness) and works with ALL routers.
        # When use_mtand is False (default) nothing below changes: the base
        # mean-pooling ECG/CXR encoders are kept and all modalities are
        # d_model wide (the original equal-width contract).
        # -----------------------------------------------------------------
        self.use_mtand = use_mtand
        self.mtand_time_scale = float(mtand_time_scale)
        if mtand_pool not in ("none", "mean", "max"):
            raise ValueError(
                f"mtand_pool must be 'none', 'mean', or 'max', got {mtand_pool!r}."
            )
        self.mtand_pool = mtand_pool

        if use_mtand:
            # Only pool='none' produces unequal (wide) widths; pooled modes keep
            # every modality at d_model, so all router types stay valid there.
            # This unequal-width restriction only applies to the POOLED placement,
            # where the static tabular vector (d_model) is routed alongside the
            # un-pooled sequences (R*d_model). The encoder placement fuses tabular
            # at the head, so its routed modalities (ECG/CXR[/TS]) are all equal
            # width and permod/disjoint are valid there.

            # Replace the base pooled ECG/CXR encoders with time-aware mTAND.
            self.ecg_encoder = MTANDSequenceEncoder(
                input_dim=ecg_dim,
                d_model=d_model,
                embed_time=mtand_embed_time,
                num_heads=mtand_num_heads,
                tt_max=mtand_tt_max,
                pool=mtand_pool,
                dropout=dropout,
            )
            self.cxr_encoder = MTANDSequenceEncoder(
                input_dim=cxr_dim,
                d_model=d_model,
                embed_time=mtand_embed_time,
                num_heads=mtand_num_heads,
                tt_max=mtand_tt_max,
                pool=mtand_pool,
                dropout=dropout,
            )
            # Widths come straight from the encoders: R*d_model when pool='none',
            # d_model when pooled. MODALITIES order is (tabular, ecg, cxr[, ts]).
            # widths = (d_model, self.ecg_encoder.output_dim, self.cxr_encoder.output_dim)
            widths = {
                "tabular": d_model,
                "ecg": self.ecg_encoder.output_dim,
                "cxr": self.cxr_encoder.output_dim,
            }
            if self.ts_temporal:
                # T3: the raw regular-TS sequence gets its own mTAND encoder with
                # concat_mask=True (upstream FuseMoE's irregular-TS convention:
                # the per-feature observation mask is concatenated onto the
                # values). Its output width matches ECG/CXR, so the equal/unequal
                # width logic below extends cleanly to 4 modalities.
                self.ts_encoder = MTANDSequenceEncoder(
                    input_dim=self.ts_dim,
                    d_model=d_model,
                    embed_time=mtand_embed_time,
                    num_heads=mtand_num_heads,
                    tt_max=mtand_tt_max,
                    pool=mtand_pool,
                    dropout=dropout,
                    concat_mask=self.ts_modality_concat_mask,
                )
                
                widths["ts"] = self.ts_encoder.output_dim
        else:
            widths = {name: d_model for name in self.modality_names}
            # if self.ts_modality_enabled and not self.ts_temporal:
            #     widths["tabular"] = d_model
            #     widths["ecg"] = d_model
            #     widths["cxr"] = d_model
            #     widths["ts"] = d_model

        self.modality_input_sizes = widths
        self.fusion_modality_input_sizes = {name: widths[name] for name in self.fusion_modality_names}
        fusion_widths = set(self.fusion_modality_input_sizes.values())
        if len(fusion_widths) != 1:
            raise ValueError(
                "All MHSA fusion modalities must share the same embedding width. "
                f"Got: {self.fusion_modality_input_sizes}"
            )
        self.fusion_input_size = int(next(iter(fusion_widths)))
        self.total_input_size = sum(
            self.modality_input_sizes[name]
            for name in self.modality_names
        )
        
        if self.perform_imputation:
            self.imputation = GCN_imputation(
                modality_input_sizes=self.modality_input_sizes,
                modality_names=self.modality_names,
                graph_construction_loss_weight=self.graph_construction_loss_weight,
                stability_loss_type=self.stability_loss_type,
                gcn_refine_mode=self.gcn_refine_mode,
                modalities_to_impute=self.modalities_to_impute,
            )
        else:
            self.imputation = None
        
        if self.use_fusion:
            self.fusion = MHSA_Fusion(
                modality_input_sizes=self.fusion_modality_input_sizes,
                modality_names=self.fusion_modality_names,
                MHD_num_head=MHD_num_head,
                dropout=dropout,
                missingness_embedding=missingness_embedding,
                mask_missingness_fusion=mask_missingness_fusion,
                fusion_dim=self.fusion_input_size,
            )
        else:
            self.fusion = None

        # Final prediction head
        self.proj1 = nn.Linear(self.total_input_size, d_model * 2)
        self.out_layer = nn.Linear(d_model * 2, output_dim)
       

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode each modality.

        Default (use_mtand=False): delegate to the base mean-pooling encoders
        (tabular/ECG/CXR all -> [B, d_model]).

        use_mtand=True: tabular stays the base flat encoder ([B, d_model]);
        ECG/CXR are encoded by time-aware mTAND, each returning either a
        flattened [B, R*d_model] sequence (mtand_pool='none') or a pooled
        [B, d_model] vector (mtand_pool='mean'/'max'). Observation times are
        divided by mtand_time_scale so they land on the mTAND reference grid
        in [0, 1].

        Static TS summary extraction is not supported.

        ts_modality on, mtand on: the raw regular-TS sequence
        (``ts_feats``/``ts_time``/``ts_missing``) is encoded by mTAND with
        concat_mask=True; ``tab_feats`` is used whole. The observation mask
        handed to mTAND is ``1 - ts_missing`` (upstream convention 1=observed).
        """
        if self.ts_temporal:
            scale = self.mtand_time_scale
            tabular = self.tabular_encoder(batch["tab_feats"], batch.get("tab_lengths"))
            ecg = self.ecg_encoder(
                batch["ecg_feats"], batch["ecg_time"] / scale, batch["ecg_lengths"]
            )
            cxr = self.cxr_encoder(
                batch["cxr_feats"], batch["cxr_time"] / scale, batch["cxr_lengths"]
            )
            ts_feats = batch["ts_feats"]
            obs_mask = 1.0 - batch["ts_missing"].to(dtype=ts_feats.dtype)
            ts = self.ts_encoder(
                ts_feats,
                batch["ts_time"] / scale,
                batch["ts_lengths"],
                obs_mask=obs_mask,
            )
            return {"tabular": tabular, "ecg": ecg, "cxr": cxr, "ts": ts}

        # if self.ts_modality_enabled:
        #     tab_feats = batch["tab_feats"]
        #     tab_lengths = batch.get("tab_lengths")
        #     tabular_block, ts_block = self._static_ts_partition(tab_feats)
        #     tabular = self.tabular_encoder(tabular_block, tab_lengths)
        #     ts = self.ts_encoder(ts_block, tab_lengths)
        #     ecg = self.ecg_encoder(batch["ecg_feats"], batch["ecg_lengths"])
        #     cxr = self.cxr_encoder(batch["cxr_feats"], batch["cxr_lengths"])
        #     return {"tabular": tabular, "ecg": ecg, "cxr": cxr, "ts": ts}

        if not self.use_mtand:
            return super().encode(batch)

        tabular = self.tabular_encoder(batch["tab_feats"], batch.get("tab_lengths"))
        ecg = self.ecg_encoder(
            batch["ecg_feats"],
            batch["ecg_time"] / self.mtand_time_scale,
            batch["ecg_lengths"],
        )
        cxr = self.cxr_encoder(
            batch["cxr_feats"],
            batch["cxr_time"] / self.mtand_time_scale,
            batch["cxr_lengths"],
        )
        return {"tabular": tabular, "ecg": ecg, "cxr": cxr}
    

    def impute(self, encoded: dict[str, torch.Tensor], modality_masks: dict[str, torch.Tensor]):
        if self.imputation is None:
            raise RuntimeError(
                "Imputation was disabled (impute=False). "
                "GCN_imputation has not been constructed."
            )
        return self.imputation(encoded, modality_masks)


    def fuse(
        self,
        encoded: dict[str, torch.Tensor],
        modality_masks: dict[str, torch.Tensor],
        return_attentions: bool = False,
    ):
        if not self.use_fusion:
            # No MHSA block is needed when only one modality is being fused.
            name = self.fusion_modality_names[0]
            out = {
                "embeds": encoded[name],
            }
            if return_attentions:
                out["fusion_attentions"] = None
            return out
        
        fusion_inputs = {name: encoded[name] for name in self.fusion_modality_names}
        fusion_masks = {name: modality_masks[name] for name in self.fusion_modality_names}

        return self.fusion(fusion_inputs, fusion_masks, return_attentions)

    def predict(self, fused_embeds: torch.Tensor) -> torch.Tensor:
        last_hs_proj = self.dropout(self.relu(self.proj1(fused_embeds)))
        outputs = self.out_layer(last_hs_proj)
        if self.output_dim == 1:
            return outputs.squeeze(-1)
        return outputs # in logit
    
    def forward(
        self,
        batch: dict[str, torch.Tensor],
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, Union[torch.Tensor, dict]]:

        modality_embeddings = self.encode(batch)
        modality_masks = self._modality_mask(batch, modality_embeddings)
        check_modality_keys(self.modality_names, modality_embeddings, modality_masks)

        if self.perform_imputation:
            encoded_embeds, sum_stability_losses, imputation_attentions = self.impute(modality_embeddings, modality_masks)
        else:
            encoded_embeds = modality_embeddings
            sum_stability_losses = None
            imputation_attentions = None

        fused_embeds = self.fuse(encoded_embeds, modality_masks, return_aux)

        unfused_modality_names = [
            name for name in self.modality_names
            if name not in self.fusion_modality_names
        ]

        unfused_embeds = [encoded_embeds[name] for name in unfused_modality_names]

        combined_embeds = torch.cat(
            unfused_embeds + [fused_embeds["embeds"]],
            dim=-1,
        )
        outputs = self.predict(combined_embeds)

        if return_aux:
            outputs_dict = {"logits": outputs}
            if self.perform_imputation:
                outputs_dict["aux_loss"] = sum_stability_losses
                outputs_dict["imputation_attentions"] = imputation_attentions
            if self.use_fusion:
                outputs_dict["fusion_attentions"] = fused_embeds.get("fusion_attentions")
            # outputs_dict["fusion_embeds_by_modality"] = fused_embeds.get("embeds_by_modality", {})
            return outputs_dict
        
        return outputs

    
    def _modality_mask(
        self,
        batch: Mapping[str, torch.Tensor],
        encoded: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        device = encoded["tabular"].device
        if not self.use_modality_mask:
            return {name: encoded[name].new_ones(encoded[name].shape[0], 1).float() for name in self.modality_names}

        if self.ts_temporal:
            # T3: TS availability is a per-sample signal -- a stay is present when
            # it has at least one real timestep. ts_missing is a per-timestep,
            # per-feature mask (not the per-sample scalar sequence_available
            # expects), so derive availability from ts_lengths directly.
            ts_lengths = batch["ts_lengths"].to(device=device)

            tab_mask = tabular_available(batch, device=device)
            ecg_mask = sequence_available(batch, "ecg", device=device)
            cxr_mask = sequence_available(batch, "cxr", device=device)
            ts_mask = ts_lengths.gt(0)
            available = {
                "tabular": to_binary_mask(tab_mask),
                "ecg": to_binary_mask(ecg_mask),
                "cxr": to_binary_mask(cxr_mask),
                "ts": to_binary_mask(ts_mask),
            }
        
        # elif self.ts_modality_enabled:
        #     # The tabular and ts modalities share `tab_missing`; partition it the
        #     # same way as the features (tabular = leading static + trailing OHE;
        #     # ts = the summary span) so each modality's availability reflects only
        #     # its own columns.
        #     tab_batch, ts_batch = self._static_tab_subbatches(batch)
        #     available = {
        #         "tabular": to_binary_mask(tabular_available(tab_batch, device=device)),
        #         "ecg": to_binary_mask(sequence_available(batch, "ecg", device=device)),
        #         "cxr": to_binary_mask(sequence_available(batch, "cxr", device=device)),
        #         "ts": to_binary_mask(tabular_available(ts_batch, device=device)),
        #     }
        
        else:
            available = {
                "tabular": to_binary_mask(tabular_available(batch, device=device)),
                "ecg": to_binary_mask(sequence_available(batch, "ecg", device=device)),
                "cxr": to_binary_mask(sequence_available(batch, "cxr", device=device)),
            }
        
        return available

    # Legacy static-TS summary partitioning.
    # Disabled because the published model supports separate TS
    # only through mTAND temporal encoding.

    # def _static_ts_partition(
    #     self,
    #     tab: torch.Tensor,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Split a tabular tensor into (tabular_block, ts_block) on the summary span.

    #     Layout of the feature axis (last dim, works for flat ``(B, F)`` and
    #     sequence ``(B, T, F)``):
    #         ``[ leading static | summary_dim summary | summary_trailing_skip OHE ]``
    #     The ts block is the middle summary span; the tabular block is the leading
    #     static columns concatenated with the trailing OHE columns (so the tabular
    #     modality keeps every non-summary feature, e.g. demographics/OHE).
    #     """
    #     summary = self.ts_modality_summary_dim
    #     skip = self.ts_modality_summary_trailing_skip
    #     if skip > 0:
    #         ts_block = tab[..., -(summary + skip) : -skip]
    #         tabular_block = torch.cat(
    #             [tab[..., : -(summary + skip)], tab[..., -skip:]], dim=-1
    #         )
    #     else:
    #         ts_block = tab[..., -summary:]
    #         tabular_block = tab[..., :-summary]
    #     return tabular_block, ts_block

    # def _static_tab_subbatches(
    #     self,
    #     batch: Mapping[str, torch.Tensor],
    # ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    #     """Return (tabular, ts) batch views for availability, partitioned like the
    #     features. ``tab_lengths`` (time axis) is carried through unchanged."""
    #     tab_view: dict[str, torch.Tensor] = {}
    #     ts_view: dict[str, torch.Tensor] = {}
    #     for key in ("tab_missing", "tab_feats"):
    #         if key in batch:
    #             tab_block, ts_block = self._static_ts_partition(batch[key])
    #             tab_view[key] = tab_block
    #             ts_view[key] = ts_block
    #     if "tab_lengths" in batch:
    #         tab_view["tab_lengths"] = batch["tab_lengths"]
    #         ts_view["tab_lengths"] = batch["tab_lengths"]
    #     return tab_view, ts_view

def _mtand_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve mTAND kwargs from a dedicated ``mtand`` sub-section.

    The preferred layout is a standalone ``mtand`` block (which train.py lifts
    from the top-level config, parallel to ``model``/``train``/``evaluation``)::

        mtand:
          enabled: true          # -> use_mtand
          pool: none             # none | mean | max
          tt_max: 64
          embed_time: 64
          num_heads: 8
          time_scale: 90.0

    When no ``mtand`` sub-section is present, fall back to the legacy flat keys
    (``use_mtand``, ``mtand_tt_max``, ...) so existing configs and saved
    checkpoints keep working unchanged.
    """
    mtand_cfg = cfg.get("mtand")
    if isinstance(mtand_cfg, Mapping):
        return {
            "use_mtand": bool(mtand_cfg.get("enabled", mtand_cfg.get("use_mtand", False))),
            "mtand_pool": str(mtand_cfg.get("pool", "none")),
            # `tt_max` (upstream name); `num_ref_points` kept as a back-compat alias.
            "mtand_tt_max": int(mtand_cfg.get("tt_max", mtand_cfg.get("num_ref_points", 64))),
            "mtand_embed_time": int(mtand_cfg.get("embed_time", 64)),
            "mtand_num_heads": int(mtand_cfg.get("num_heads", 8)),
            "mtand_time_scale": float(mtand_cfg.get("time_scale", 90.0)),
        }
    return {
        "use_mtand": bool(cfg.get("use_mtand", False)),
        "mtand_pool": str(cfg.get("mtand_pool", "none")),
        "mtand_tt_max": int(cfg.get("mtand_tt_max", cfg.get("mtand_num_ref_points", 64))),
        "mtand_embed_time": int(cfg.get("mtand_embed_time", 64)),
        "mtand_num_heads": int(cfg.get("mtand_num_heads", 8)),
        "mtand_time_scale": float(cfg.get("mtand_time_scale", 90.0)),
    }


def _ts_modality_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the optional raw time-series modality settings.

    A separate TS modality requires mTAND temporal encoding.
    Static TS summary extraction is not supported.

    Supported configuration:

        ts_modality:
        enabled: false
        concat_mask: true

    When absent, the TS modality is disabled.
    """
    off = {
        "ts_modality_enabled": False,
        # "ts_modality_summary_dim": None,
        # "ts_modality_summary_trailing_skip": 0,
        "ts_modality_concat_mask": True,
    }
    raw = cfg.get("ts_modality")
    if raw is None:
        return dict(off)
    if isinstance(raw, Mapping):
        enabled = bool(raw.get("enabled", raw.get("on", False)))
        # summary_dim = raw.get("summary_dim")
        return {
            "ts_modality_enabled": enabled,
            # "ts_modality_summary_dim": (
            #     int(summary_dim) if summary_dim is not None else None
            # ),
            # "ts_modality_summary_trailing_skip": int(
            #     raw.get("summary_trailing_skip", 0)
            # ),
            "ts_modality_concat_mask": bool(raw.get("concat_mask", True)),
        }
    if isinstance(raw, bool):
        return {**off, "ts_modality_enabled": raw}
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in ("off", "false", "no", "0", "tabular", "summary"):
            return dict(off)
        if token in ("on", "true", "yes", "1", "modality"):
            return {**off, "ts_modality_enabled": True}
    raise ValueError(
        "ts_modality must be a mapping with an 'enabled' bool, a bool, or one of "
        "'off'/'on' (aliases: tabular/summary, modality); got "
        f"{raw!r}."
    )


def build_mtand_m3care_model_from_config(
    fusion_cfg: Mapping[str, Any],
    feature_dims: Mapping[str, int],
    output_dim: int = 1,
) -> mTAND_M3CareModel:
    """Build the mTAND-M3Care model from fusion config and dataset feature dimensions."""
    print(f"Dimensions of modality embeddings: {feature_dims}")
    modalities_to_impute = fusion_cfg.get("modalities_to_impute", None)
    if modalities_to_impute is not None:
        modalities_to_impute = list(modalities_to_impute)
    
    # Config for inlcuding a 4th modality of time series
    ts_settings = _ts_modality_settings(fusion_cfg)

    # Raw regular-TS width for the temporal (T3) path; None for bundles without
    # ts_feats (the model only reads it when ts_modality + mtand are both on).
    ts_dim = feature_dims.get("ts")

    # mTAND settings
    mtand_settings = _mtand_settings(fusion_cfg)

    modality_names = list(fusion_cfg.get("modality_names", MODALITIES[:3]))
    if ts_settings["ts_modality_enabled"] and "ts" not in modality_names:
        modality_names.append("ts")

    fusion_modality_names = fusion_cfg.get("fusion_modality_names", None)
    if fusion_modality_names is None:
        fusion_modality_names = [name for name in modality_names if name != "tabular"]
    else:
        fusion_modality_names = list(fusion_modality_names)

    return mTAND_M3CareModel(
        tabular_dim=int(feature_dims["tabular"]),
        ecg_dim=int(feature_dims["ecg"]),
        cxr_dim=int(feature_dims["cxr"]),
        ts_dim=int(feature_dims["ts"]) if feature_dims.get("ts") is not None else None,
        d_model=int(fusion_cfg.get("d_model", 128)),
        modality_names=modality_names,
        fusion_modality_names=fusion_modality_names,
        output_dim=int(fusion_cfg.get("output_dim", output_dim)),
        dropout=float(fusion_cfg.get("dropout", 0.0)),
        MHD_num_head=int(fusion_cfg.get("MHD_num_head", 4)),
        use_modality_mask=bool(fusion_cfg.get("use_modality_mask", True)),
        impute=bool(fusion_cfg.get("impute", False)),
        missingness_embedding=bool(fusion_cfg.get("missingness_embedding", False)),
        graph_construction_loss_weight=float(fusion_cfg.get("graph_construction_loss_weight", 0.0)),
        stability_loss_type=str(fusion_cfg.get("stability_loss_type", "norm")),
        gcn_refine_mode=str(fusion_cfg.get("gcn_refine_mode", "adaptive")),
        mask_missingness_fusion=bool(fusion_cfg.get("mask_missingness_fusion", False)),
        modalities_to_impute=modalities_to_impute,
        **mtand_settings,
        **ts_settings,
    )






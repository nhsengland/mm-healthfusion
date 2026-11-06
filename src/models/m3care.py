"""M3Care model that supports contrastive learning.

- Observed-modality self-alignment: InfoNCE loss
- Missing-modality cross-modal alignment: InfoNCE loss
- Learned cross-modal target supervision
  - Reconstruction loss: MSE or cosine (default)
  - Optional symmetric InfoNCE loss
"""

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
from src.data.collate import modality_available

MODALITIES = ("tabular", "ecg", "cxr")

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
    # def __init__(self, input_dim, d_model, MHD_num_head, d_ff, output_dim, dropout=0.5):
    def __init__(self, input_dim, d_model, MHD_num_head, d_ff, dropout=0.5):
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
        feature_dim = input.size(2)
        assert(feature_dim == self.input_dim)
        assert(self.d_model % self.MHD_num_head == 0)

     
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
    Generic multimodal imputation via batch-wise GCN for pre-extracted embeddings.

    Expects:
      - modality_embeddings: dict[str, Tensor] with each Tensor shaped (B, H)
      - modality_masks:      dict[str, Tensor] with each Tensor shaped (B, 1)

    Returns:
      - dict of refined embeddings keyed by modality name
      - stability_loss scalar tensor
      - contrastive losses dict
      - imputation attentions per modality
    """
    def __init__(
        self,
        hidden_dim: int,
        modality_names: tuple[str, ...] | list[str],
        graph_construction_loss_weight: float = 0.01,
        stability_loss_type: str = "norm",
        mean_graph_construction_loss: bool = False,
        gcn_refine_mode: str = "adaptive",
        modalities_to_impute: tuple[str, ...] | list[str] = None,
        contrastive_modalities: tuple[str, ...] | list[str] | None = None,
        enable_observed_contrastive: bool = False,
        enable_missing_contrastive: bool = False,
        observed_contrastive_weight: float = 1.0,
        missing_contrastive_weight: float = 1.0,
        contrastive_temperature: float = 0.1,
        contrastive_projection_dim: int | None = None,
        cross_modal_target_modalities: tuple[str, ...] | list[str] | None = None,
        min_contrastive_samples: int = 2,
        learned_cross_modal_target: bool = False,
        enable_learned_target_supervision: bool = False,
        learned_target_loss_weight: float = 1.0,
        learned_target_loss_type: str = "cosine",
        learned_target_contrastive_weight: float = 1.0,
        learned_target_contrastive_temperature: float | None = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.modality_names = list(modality_names)
        self.modal_num = len(self.modality_names)
        self.graph_construction_loss_weight = graph_construction_loss_weight
        self.stability_loss_type = stability_loss_type.lower().strip()
        self.mean_graph_construction_loss = bool(mean_graph_construction_loss)
        self.gcn_refine_mode = gcn_refine_mode.lower().strip()

        # Generic contrastive-learning configuration.
        self.enable_observed_contrastive = bool(enable_observed_contrastive)
        self.enable_missing_contrastive = bool(enable_missing_contrastive)
        self.observed_contrastive_weight = float(observed_contrastive_weight)
        self.missing_contrastive_weight = float(missing_contrastive_weight)
        self.contrastive_temperature = float(contrastive_temperature)
        self.min_contrastive_samples = int(min_contrastive_samples)
        self.learned_cross_modal_target = bool(learned_cross_modal_target)
        self.enable_learned_target_supervision = bool(enable_learned_target_supervision)
        self.learned_target_loss_weight = float(learned_target_loss_weight)
        self.learned_target_loss_type = str(learned_target_loss_type).lower().strip()
        self.learned_target_contrastive_weight = float(learned_target_contrastive_weight)
        self.learned_target_contrastive_temperature = (
            self.contrastive_temperature
            if learned_target_contrastive_temperature is None
            else float(learned_target_contrastive_temperature)
        )
        if self.contrastive_temperature <= 0.0:
            raise ValueError("contrastive_temperature must be positive.")
        if self.min_contrastive_samples < 2:
            raise ValueError( "min_contrastive_samples must be at least 2.")
        if self.learned_target_loss_weight < 0.0:
            raise ValueError("learned_target_loss_weight must be non-negative.")
        if self.learned_target_contrastive_weight < 0.0:
            raise ValueError("learned_target_contrastive_weight must be non-negative.")
        if self.learned_target_contrastive_temperature <= 0.0:
            raise ValueError("learned_target_contrastive_temperature must be positive.")
        if self.learned_target_loss_type not in {"cosine", "mse"}:
            raise ValueError(
                "learned_target_loss_type must be either 'cosine' or 'mse'."
            )
        if self.enable_learned_target_supervision and not self.learned_cross_modal_target:
            raise ValueError(
                "enable_learned_target_supervision=True requires "
                "learned_cross_modal_target=True."
            )

        if self.gcn_refine_mode not in {"adaptive", "imputed_only", "imputed_for_missingness"}:
            raise ValueError(
                "gcn_refine_mode must be one of: 'adaptive', 'imputed_only', 'imputed_for_missingness'."
            )

        if self.stability_loss_type not in {"norm", "cosine", "similarity", "matrix", "similarity_matrix"}:
            raise ValueError(
                f"Unknown stability_loss_type='{self.stability_loss_type}'. "
                "stability_loss_type must be one of: "
                "'norm', 'cosine', 'similarity', 'matrix', 'similarity_matrix'."
            )

        if modalities_to_impute is None:
            # Default: impute all modalities
            self.modalities_to_impute = self.modality_names
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

        if contrastive_modalities is None:
            # Default: apply contrastive objectives to all imputed modalities.
            self.contrastive_modalities = list(self.modalities_to_impute)
            print(
                "Since contrastive_modalities is not specified, "
                f"default to: {self.contrastive_modalities}"
            )
        elif len(contrastive_modalities) == 0:
            raise ValueError(
                "contrastive_modalities cannot be empty when contrastive learning is enabled."
            )
        elif set(contrastive_modalities).issubset(self.modality_names):
            self.contrastive_modalities = [
                m for m in self.modality_names if m in contrastive_modalities
            ]
        else:
            invalid = sorted(set(contrastive_modalities).difference(self.modality_names))
            raise ValueError(
                f"Unknown modalities in contrastive_modalities: {invalid}. "
                f"Available modalities are: {self.modality_names}."
            )

        if (
            self.enable_observed_contrastive
            or self.enable_missing_contrastive
            or self.enable_learned_target_supervision
        ):
            if not set(self.contrastive_modalities).issubset(set(self.modalities_to_impute)):
                invalid = sorted(set(self.contrastive_modalities).difference(self.modalities_to_impute))
                raise ValueError(
                    "Contrastive modalities must be included in modalities_to_impute. "
                    f"Invalid entries: {invalid}."
                )
        else:
            print(
                "Contrastive learning is not performed, ",
                "despite the configuration of the contrastive_modalities."
            )

        print(
            "M3Care modality configuration:",
            {
                "modality_names": self.modality_names,
                "modalities_to_impute": self.modalities_to_impute,
                "contrastive_modalities": self.contrastive_modalities,
            },
        )


        if cross_modal_target_modalities is None:
            # Default: use all non-target modalities as the context target.
            self.cross_modal_target_modalities = None
        elif len(cross_modal_target_modalities) == 0:
            raise ValueError(
                "cross_modal_target_modalities cannot be empty when missing contrastive loss is enabled."
            )
        elif set(cross_modal_target_modalities).issubset(self.modality_names):
            self.cross_modal_target_modalities = [
                m for m in self.modality_names if m in cross_modal_target_modalities
            ]
        else:
            invalid = sorted(set(cross_modal_target_modalities).difference(self.modality_names))
            raise ValueError(
                f"Unknown modalities in cross_modal_target_modalities: {invalid}. "
                f"Available modalities are: {self.modality_names}."
            )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(hidden_dim)

        if (
            self.enable_observed_contrastive
            or self.enable_missing_contrastive
            or self.enable_learned_target_supervision
        ):
            proj_dim = int(contrastive_projection_dim) if contrastive_projection_dim is not None else int(hidden_dim)
            self.contrastive_projection_dim = proj_dim
            # Anchor projector: maps GCN-imputed embeddings into contrastive space
            # Updated by the contrastive loss through the GCN/anchor branch.
            self.contrastive_projector = nn.Sequential(
                nn.Linear(hidden_dim, proj_dim, bias=True),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim, bias=True),
            )
            # Target projector: maps target/teacher embeddings into contrastive space.
            # Starts identical to the anchor projector. It is trained
            # during Stage 1, then frozen together with the modality encoders when
            # freeze_modality_encoders=True for Stage 2.
            self.contrastive_target_projector = copy.deepcopy(
                self.contrastive_projector
            )

        # Learned cross-modal target networks are separate from the contrastive
        # target projector. Each target modality gets an MLP that predicts its
        # embedding from the selected non-target context modalities. This is
        # trained on complete/observed cases in Stage 1 and frozen in Stage 2.

        if (
            self.learned_cross_modal_target
            and (
                self.enable_missing_contrastive
                or self.enable_learned_target_supervision
            )
        ):
            self.learned_target_networks = nn.ModuleDict()
            for target_modality in self.contrastive_modalities:
                context_modalities = self._resolve_context_modalities(target_modality)
                if not context_modalities:
                    raise ValueError(
                        f"No context modalities are available for target modality "
                        f"{target_modality!r}."
                    )
                context_dim = len(context_modalities) * hidden_dim + len(context_modalities)
                self.learned_target_networks[target_modality] = nn.Sequential(
                    nn.Linear(context_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )

        # Similarity projection heads (one per modality)
        self.simiProj = clones(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=True),
            ),
            self.modal_num,
        )

        # Similarity projection heads (one per modality)
        self.GCN = clones(GraphConvolution(hidden_dim, hidden_dim, bias=True), self.modal_impute_num)
        self.GCN_2 = clones(GraphConvolution(hidden_dim, hidden_dim, bias=True), self.modal_impute_num)

        # Learnable mixing parameters
        self.eps_params = nn.ParameterList(
            [nn.Parameter(torch.ones(size=(1,))) for _ in range(self.modal_num)]
        )
        self.threshold = nn.Parameter(torch.zeros(size=(1,)))

        # Per-modality gating for refined vs original embeddings
        if self.gcn_refine_mode == "adaptive":
            # Weights only for modalities selected for imputation
            self.weight1 = clones(nn.Linear(hidden_dim, 1), self.modal_impute_num)
            self.weight2 = clones(nn.Linear(hidden_dim, 1), self.modal_impute_num)
        else:
            # Since no adaptive fusion will be performed, the weights are set to None as placeholder
            self.weight1 = [None] * self.modal_impute_num
            self.weight2 = [None] * self.modal_impute_num
    
    def set_contrastive_target_projector_trainable(
        self,
        trainable: bool = True,
    ) -> list[str]:
        """
        Freeze/unfreeze the contrastive target projector.

        Returns the names of parameters whose requires_grad state was updated.
        """
        updated: list[str] = []
        for name, param in self.contrastive_target_projector.named_parameters():
            param.requires_grad_(trainable)
            updated.append(f"contrastive_target_projector.{name}")
        return updated
    
    def apply_frozen_contrastive_target_projector_state(
        self,
    ) -> list[str]:
        """
        Freeze the contrastive target projector when modality encoders are frozen.

        The target projector is put into eval mode so that any future
        train/eval-sensitive layers are also kept fixed in behaviour.
        """
        updated = self.set_contrastive_target_projector_trainable(False)
        self.contrastive_target_projector.eval()
        return updated

    def set_learned_target_networks_trainable(
        self,
        trainable: bool = True,
    ) -> list[str]:
        """Enable or disable gradients for learned cross-modal target networks."""
        updated: list[str] = []
        if not hasattr(self, "learned_target_networks"):
            return updated
        for target_modality, network in self.learned_target_networks.items():
            for name, param in network.named_parameters():
                param.requires_grad_(trainable)
                updated.append(f"learned_target_networks.{target_modality}.{name}")
        return updated

    def apply_frozen_learned_target_network_state(self) -> list[str]:
        """Freeze learned cross-modal target networks and keep them in eval mode."""
        updated = self.set_learned_target_networks_trainable(False)
        if hasattr(self, "learned_target_networks"):
            self.learned_target_networks.eval()
        return updated
    
    def _project_contrastive(
        self,
        x: torch.Tensor,
        *,
        projector: str = "anchor",
    ) -> torch.Tensor:
        """Project an embedding into the anchor or target contrastive space."""
        if projector == "anchor":
            projection_head = self.contrastive_projector
        elif projector == "target":
            projection_head = self.contrastive_target_projector
        else:
            raise ValueError(
                "projector must be either 'anchor' or 'target', "
                f"got {projector!r}."
            )

        return projection_head(x)

    def _pairwise_info_nce(self, anchor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Symmetric InfoNCE loss with batch-wise positives on the diagonal.
        """
        if anchor.ndim != 2 or target.ndim != 2:
            raise ValueError("Contrastive inputs must have shape (B, H).")
        if anchor.size(0) != target.size(0):
            raise ValueError("Contrastive inputs must have the same batch size.")

        batch_size = anchor.size(0)
        if batch_size < self.min_contrastive_samples:
            return anchor.new_tensor(0.0)

        anchor = F.normalize(
            self._project_contrastive(anchor, projector="anchor"),
            p=2,
            dim=-1,
            eps=1e-8,
        )
        
        # Always block contrastive gradients from reaching the target embedding
        # itself. During Stage 1 the target projector remains trainable, so it
        # can learn the target representation. Once freeze_modality_encoders=True
        # is applied, the target projector is frozen as well and the whole target
        # branch is evaluated without autograd during Stage 2.
        target = target.detach()
        target_projector_trainable = any(
            parameter.requires_grad
            for parameter in self.contrastive_target_projector.parameters()
        )

        if target_projector_trainable:
            target = self._project_contrastive(target, projector="target")
        else:
            with torch.no_grad():
                target = self._project_contrastive(target, projector="target")

        target = F.normalize(
            target,
            p=2,
            dim=-1,
            eps=1e-8,
        )

        logits = torch.matmul(anchor, target.t()) / self.contrastive_temperature
        labels = torch.arange(batch_size, device=anchor.device)
        loss_a2t = F.cross_entropy(logits, labels)
        loss_t2a = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_a2t + loss_t2a)

    def _resolve_context_modalities(self, target_modality: str) -> list[str]:
        if self.cross_modal_target_modalities is None:
            candidates = [m for m in self.modality_names if m != target_modality]
        else:
            candidates = [m for m in self.cross_modal_target_modalities if m != target_modality]
            if len(candidates) == 0:
                candidates = [m for m in self.modality_names if m != target_modality]
        return candidates

    def _build_context_features(
        self,
        modality_embeddings: Mapping[str, torch.Tensor],
        modality_masks: Mapping[str, torch.Tensor],
        target_modality: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build mask-aware concatenated context features for one target modality."""
        context_modalities = self._resolve_context_modalities(target_modality)
        context_embeds: list[torch.Tensor] = []
        context_masks: list[torch.Tensor] = []

        for name in context_modalities:
            emb = modality_embeddings[name]
            mask = modality_masks[name].to(device=emb.device, dtype=emb.dtype)
            if emb.ndim != 2:
                raise ValueError("Modality embeddings must have shape (B, H).")
            if mask.ndim != 2 or mask.shape[1] != 1:
                raise ValueError("Modality masks must have shape (B, 1).")
            context_embeds.append(emb * mask)
            context_masks.append(mask)

        features = torch.cat(context_embeds + context_masks, dim=-1)
        valid_mask = torch.stack(context_masks, dim=0).sum(dim=0).squeeze(1) > 0
        return features, valid_mask

    def _build_average_cross_modal_target(
        self,
        modality_embeddings: Mapping[str, torch.Tensor],
        modality_masks: Mapping[str, torch.Tensor],
        target_modality: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the baseline mean-of-available-context target."""
        context_modalities = self._resolve_context_modalities(target_modality)
        context_embeds: list[torch.Tensor] = []
        context_masks: list[torch.Tensor] = []

        for name in context_modalities:
            emb = modality_embeddings[name]
            mask = modality_masks[name].to(device=emb.device, dtype=emb.dtype)
            if emb.ndim != 2:
                raise ValueError("Modality embeddings must have shape (B, H).")
            if mask.ndim != 2 or mask.shape[1] != 1:
                raise ValueError("Modality masks must have shape (B, 1).")
            context_embeds.append(emb * mask)
            context_masks.append(mask)

        stacked_embeds = torch.stack(context_embeds, dim=0).sum(dim=0)
        stacked_masks = torch.stack(context_masks, dim=0).sum(dim=0)
        valid_mask = stacked_masks.squeeze(1) > 0
        context_target = stacked_embeds / stacked_masks.clamp_min(1.0)
        return context_target, valid_mask

    def _build_cross_modal_target(
        self,
        modality_embeddings: Mapping[str, torch.Tensor],
        modality_masks: Mapping[str, torch.Tensor],
        target_modality: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build either the baseline mean target or the learned target predictor."""
        if not self.learned_cross_modal_target:
            return self._build_average_cross_modal_target(
                modality_embeddings,
                modality_masks,
                target_modality,
            )

        features, valid_mask = self._build_context_features(
            modality_embeddings,
            modality_masks,
            target_modality,
        )
        context_target = self.learned_target_networks[target_modality](features)
        return context_target, valid_mask

    def _compute_learned_target_supervision_loss(
        self,
        predicted_target: torch.Tensor,
        true_target: torch.Tensor,
    ) -> torch.Tensor:
        """Align a learned cross-modal target prediction to the true target embedding."""
        if self.learned_target_loss_type == "mse":
            return F.mse_loss(predicted_target, true_target.detach())
        
        if self.learned_target_loss_type == "cosine":
            pred_norm = F.normalize(predicted_target, p=2, dim=-1, eps=1e-8)
            target_norm = F.normalize(true_target.detach(), p=2, dim=-1, eps=1e-8)
            return (1.0 - (pred_norm * target_norm).sum(dim=-1)).mean()

    def _compute_learned_target_contrastive_loss(
        self,
        predicted_target: torch.Tensor,
        true_target: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetric InfoNCE that makes learned targets patient-discriminative.

        The learned target prediction is the trainable anchor and is projected
        through the target projector so that both predicted and true targets lie
        in the same target/true contrastive space. The true target is
        detached and evaluated without gradients.
        """
        if predicted_target.ndim != 2 or true_target.ndim != 2:
            raise ValueError(
                "Learned-target contrastive inputs must have shape (B, H)."
            )
        if predicted_target.size(0) != true_target.size(0):
            raise ValueError(
                "Learned-target contrastive inputs must have the same batch size."
            )

        batch_size = predicted_target.size(0)
        if batch_size < self.min_contrastive_samples:
            return predicted_target.new_tensor(0.0)
        
        # Use the target projector for both the predicted target and the true modality representation
        # Asks the learned target network specifically to produce an embedding that 
        # the target projector interprets like a real embedding, which matches the Stage-2
        # use of the learned target.
        # Both branches are projected into the target/true representation space.
        # The true-target branch is detached/frozen; gradients from the predicted
        # branch can update the learned target network and target projector in Stage 1.
        anchor = F.normalize(
            self._project_contrastive(predicted_target, projector="target"),
            p=2,
            dim=-1,
            eps=1e-8,
        )

        with torch.no_grad():
            target = self._project_contrastive(
                true_target.detach(),
                projector="target",
            )
            target = F.normalize(target, p=2, dim=-1, eps=1e-8)

        logits = anchor @ target.T / self.learned_target_contrastive_temperature
        labels = torch.arange(batch_size, device=anchor.device)
        loss_a2t = F.cross_entropy(logits, labels)
        loss_t2a = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_a2t + loss_t2a)

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

    def _build_similarity(self, x, proj, eps_param):
        """
        x: (B, H)
        Returns:
          sim: batch similarity matrix (B, B)
          stability_loss: scalar
        """
        proj_x = proj(x)
        proj_x_for_graph = self.bn(self.relu(proj_x))
        x_for_graph = self.bn(x)

        sim1 = gaussian_kernel(proj_x_for_graph, kernel_mul=2.0, kernel_num=3)
        sim2 = gaussian_kernel(x_for_graph, kernel_mul=2.0, kernel_num=3)

        gate = self.sigmoid(eps_param)
        sim = ((1 - gate) * sim1 + gate) * sim2

        stability_loss = self._compute_stability_loss(x, proj_x, proj_x_for_graph)

        return sim, stability_loss

    def _refine_with_gcn(self, sim_score, x, mask, gcn1, gcn2, weight1, weight2):
        """
        sim_score: (B, B)
        x: (B, H)
        """
        x0 = self.relu(gcn1(sim_score, x))
        x0 = self.relu(gcn2(sim_score, x0))

        # Force missing-modality samples to use x0 only
        mask = mask.to(device=x.device, dtype=x.dtype)

        if self.gcn_refine_mode == "adaptive":
            # M3care:
            # - present samples: adaptive mix of x0 and x
            # - missing samples: x0 only
            w1 = torch.sigmoid(weight1(x0))
            w2 = torch.sigmoid(weight2(x))
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

    def _compute_modality_contrastive_losses(
        self,
        target_modality: str,
        modality_embeddings: Mapping[str, torch.Tensor],
        modality_masks: Mapping[str, torch.Tensor],
        imputed_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        if target_modality not in self.contrastive_modalities:
            return losses

        modality_mask = modality_masks[target_modality].to(
            device=imputed_embedding.device
        ).squeeze(1).bool()
        original_embedding = modality_embeddings[target_modality]
        batch_size = int(modality_mask.numel())

        if self.enable_learned_target_supervision and self.learned_cross_modal_target:
            learned_target, learned_target_valid = self._build_cross_modal_target(
                modality_embeddings,
                modality_masks,
                target_modality=target_modality,
            )
            supervision_mask = modality_mask & learned_target_valid
            supervision_count = int(supervision_mask.sum().item())
            supervision_fraction = supervision_count / max(batch_size, 1)

            if supervision_count > 0:
                losses[f"learned_target_{target_modality}_reconstruction_sample_count"] = (
                    imputed_embedding.new_tensor(float(supervision_count))
                    )
                losses[f"learned_target_{target_modality}_reconstruction_sample_fraction"] = (
                    imputed_embedding.new_tensor(float(supervision_fraction))
                )
                pred = learned_target[supervision_mask]
                true = original_embedding[supervision_mask]
                losses[f"learned_target_{target_modality}_reconstruction_loss"] = (
                    self._compute_learned_target_supervision_loss(pred, true)
                )
            else:
                losses[f"learned_target_{target_modality}_reconstruction_sample_count"] = imputed_embedding.new_tensor(0.0)
                losses[f"learned_target_{target_modality}_reconstruction_sample_fraction"] = imputed_embedding.new_tensor(0.0)
                losses[f"learned_target_{target_modality}_reconstruction_loss"] = imputed_embedding.new_tensor(0.0)

            if supervision_count >= self.min_contrastive_samples:
                losses[f"learned_target_{target_modality}_contrastive_sample_count"] = (
                    imputed_embedding.new_tensor(float(supervision_count))
                    )
                losses[f"learned_target_{target_modality}_contrastive_sample_fraction"] = (
                    imputed_embedding.new_tensor(float(supervision_fraction))
                )
                losses[f"learned_target_{target_modality}_contrastive_loss"] = (
                    self._compute_learned_target_contrastive_loss(
                        learned_target[supervision_mask],
                        original_embedding[supervision_mask],
                    )
                )
            else:
                losses[f"learned_target_{target_modality}_contrastive_sample_count"] = imputed_embedding.new_tensor(0.0)
                losses[f"learned_target_{target_modality}_contrastive_sample_fraction"] = imputed_embedding.new_tensor(0.0)
                losses[f"learned_target_{target_modality}_contrastive_loss"] = imputed_embedding.new_tensor(0.0)

        if self.enable_observed_contrastive:
            observed_mask = modality_mask
            observed_count = int(observed_mask.sum().item())
            losses[f"observed_{target_modality}_contrastive_sample_count"] = imputed_embedding.new_tensor(float(observed_count))
            losses[f"observed_{target_modality}_contrastive_sample_fraction"] = imputed_embedding.new_tensor(
                observed_count / max(batch_size, 1)
            )
            if observed_count >= self.min_contrastive_samples:
                losses[f"observed_{target_modality}_contrastive_loss"] = self._pairwise_info_nce(
                    imputed_embedding[observed_mask],
                    original_embedding[observed_mask].detach(),
                )
            else:
                losses[f"observed_{target_modality}_contrastive_loss"] = imputed_embedding.new_tensor(0.0)

        if self.enable_missing_contrastive:
            missing_mask = ~modality_mask
            context_target, context_valid_mask = self._build_cross_modal_target(
                modality_embeddings,
                modality_masks,
                target_modality=target_modality,
            )
            selected_mask = missing_mask & context_valid_mask
            missing_count = int(selected_mask.sum().item())
            losses[f"missing_{target_modality}_contrastive_sample_count"] = imputed_embedding.new_tensor(float(missing_count))
            losses[f"missing_{target_modality}_contrastive_sample_fraction"] = imputed_embedding.new_tensor(
                missing_count / max(batch_size, 1)
            )
            if missing_count >= self.min_contrastive_samples:
                losses[f"missing_{target_modality}_contrastive_loss"] = self._pairwise_info_nce(
                    imputed_embedding[selected_mask],
                    context_target[selected_mask].detach(),
                )
            else:
                losses[f"missing_{target_modality}_contrastive_loss"] = imputed_embedding.new_tensor(0.0)

        return losses

    def forward(
        self,
        modality_embeddings: Mapping[str, torch.Tensor],
        modality_masks: Mapping[str, torch.Tensor]
    ):

        check_modality_keys(self.modality_names, modality_embeddings, modality_masks)

        sim_list = []
        mask_list = []
        stability_losses = []

        for idx, name in enumerate(self.modality_names):
            x = modality_embeddings[name]
            mask = modality_masks[name]

            # 1) Build pairwise missing-modality masks: (B, B)
            mask2 = pairwise_mask(mask)

            # 2) Build pairwise similarity matrices for each modality
            sim, stability_loss = self._build_similarity(x, self.simiProj[idx], self.eps_params[idx])
            sim = sim * mask2 # Remove graph edges involving missingness
            sim_list.append(sim)
            mask_list.append(mask2)
            stability_losses.append(stability_loss)

        # 3) Compute combined similarity score for inputs in the batch and the sum/mean of stability losses
        sum_masks = torch.stack(mask_list, dim=0).sum(dim=0).clamp_min(1.0)
        similar_score = torch.stack(sim_list, dim=0).sum(dim=0) / sum_masks
        stability_loss_tensor = torch.stack(stability_losses, dim=0)

        if self.mean_graph_construction_loss:
            graph_construction_loss = stability_loss_tensor.mean()
        else:
            graph_construction_loss = stability_loss_tensor.sum()

        # 4) Thresholding similar to original M3Care
        threshold = self.sigmoid(self.threshold)[0]
        similar_score = self.relu(similar_score - threshold)
        bin_mask = similar_score > 0
        similar_score = similar_score + bin_mask * threshold.detach()

        # 5) Graph convolution refinement + residual mixing
        refined_outputs = {}
        imputation_attentions = {}
        contrastive_losses = {}
        # Perform imputation for selected modalities
        impute_modality_idx = 0
        for name in self.modality_names:
            if name in self.modalities_to_impute:
                x0, x_final, attns = self._refine_with_gcn(
                    similar_score,
                    modality_embeddings[name],
                    modality_masks[name],
                    self.GCN[impute_modality_idx],
                    self.GCN_2[impute_modality_idx],
                    self.weight1[impute_modality_idx],
                    self.weight2[impute_modality_idx],
                )
                refined_outputs[name] = x_final
                imputation_attentions[name] = attns

                contrastive_losses.update(
                    self._compute_modality_contrastive_losses(
                        target_modality=name,
                        modality_embeddings=modality_embeddings,
                        modality_masks=modality_masks,
                        imputed_embedding=x0,
                    )
                )

                impute_modality_idx += 1
            else:
                refined_outputs[name] = modality_embeddings[name]

        return refined_outputs, graph_construction_loss, contrastive_losses, imputation_attentions

class MHSA_Fusion(nn.Module):
    """
    Generic multimodal fusion model for pre-extracted embeddings.

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
        hidden_dim: int,
        modality_names: tuple[str, ...] | list[str],
        MHD_num_head: int = 4,
        dropout: float = 0.0,
        missingness_embedding: bool = False,
        mask_missingness_fusion: bool = False,
    ):
        super().__init__()

        self.modality_names = list(modality_names)
        self.num_modalities = len(self.modality_names)
        self.missingness_embedding = missingness_embedding
        self.mask_missingness_fusion = mask_missingness_fusion

        self.token_type_embeddings = nn.Embedding(self.num_modalities, hidden_dim)
        self.token_type_embeddings.apply(init_weights)

        if self.missingness_embedding:
            self.missingness_embeddings = nn.Embedding(2, hidden_dim)
            self.missingness_embeddings.apply(init_weights)

        self.MM_model = MM_transformer_encoder(
            input_dim=hidden_dim,
            d_model=hidden_dim,
            MHD_num_head=MHD_num_head,
            d_ff=hidden_dim * MHD_num_head,
            dropout=dropout,
        )

        self.MM_model2 = MM_transformer_encoder(
            input_dim=hidden_dim,
            d_model=hidden_dim,
            MHD_num_head=1,
            d_ff=hidden_dim * MHD_num_head,
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
        fused_tokens = [z2[:, i, :] for i in range(self.num_modalities)]

        # 5) Embedding concatenation
        combined_hidden = torch.cat(fused_tokens, dim=-1)

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

        return {"embeds": combined_hidden}


class M3CareModel(BaseMultimodalModel):
    """
    Generic multimodal fusion model for pre-extracted embeddings.
    The number and names of modalities are determined by `modality_names`.

    It:
      1) projects each modality to a shared hidden space
      2) builds pairwise similarity matrices within the batch
      3) computes graph similarity / adjacency matrices
      4) applies graph convolution to selected modalities
      5) mixes refined and original embeddings
      6) fuses all configured modality tokens with a Transformer
      7) predicts the final output
    """
    supports_aux_output = True

    def __init__(
            self,
            tabular_dim: int | None = None,
            ecg_dim: int | None = None,
            cxr_dim: int | None = None,
            d_model: int = 128,
            modality_names: tuple[str, ...] | list[str] | None = None,
            output_dim: int = 1,
            dropout: float = 0.0,
            MHD_num_head: int = 4,
            use_modality_mask: bool = False,
            impute: bool = True,
            missingness_embedding: bool = False,
            graph_construction_loss_weight: float = 0,
            stability_loss_type: str = "norm",
            mean_graph_construction_loss: bool = False,
            gcn_refine_mode: str = "adaptive",
            mask_missingness_fusion: bool = False,
            modalities_to_impute: tuple[str, ...] | list[str] = None,
            feature_dims: dict[str, int] | None = None,
            contrastive_modalities: tuple[str, ...] | list[str] | None = None,
            enable_observed_contrastive: bool = False,
            enable_missing_contrastive: bool = False,
            observed_contrastive_weight: float = 1.0,
            missing_contrastive_weight: float = 1.0,
            contrastive_temperature: float = 0.1,
            contrastive_projection_dim: int | None = None,
            cross_modal_target_modalities: tuple[str, ...] | list[str] | None = None,
            min_contrastive_samples: int = 2,
            learned_cross_modal_target: bool = False,
            enable_learned_target_supervision: bool = False,
            learned_target_loss_weight: float = 1.0,
            learned_target_loss_type: str = "cosine",
            learned_target_contrastive_weight: float = 1.0,
            learned_target_contrastive_temperature: float | None = None,
            freeze_modality_encoders: bool = False,
    ) -> None:
        modality_names = list(modality_names or MODALITIES)
        super().__init__(
            tabular_dim=tabular_dim,
            ecg_dim=ecg_dim,
            cxr_dim=cxr_dim,
            d_model=d_model,
            output_dim=output_dim,
            dropout=dropout,
            feature_dims=feature_dims,
            modality_names=modality_names,
        )
        self.hidden_dim = d_model
        self.output_dim = output_dim
        self.dropout_prob = dropout
        self.modal_num = len(self.modality_names)
        self.perform_imputation = impute
        self.graph_construction_loss_weight = graph_construction_loss_weight
        self.stability_loss_type = stability_loss_type.lower().strip()
        self.mean_graph_construction_loss = bool(mean_graph_construction_loss)
        self.gcn_refine_mode = gcn_refine_mode.lower().strip()
        self.modalities_to_impute = modalities_to_impute
        self.freeze_modality_encoders = bool(freeze_modality_encoders)

        # Generic switches
        self.enable_observed_contrastive = bool(enable_observed_contrastive)
        self.enable_missing_contrastive = bool(enable_missing_contrastive)
        self.observed_contrastive_weight = float(observed_contrastive_weight)
        self.missing_contrastive_weight = float(missing_contrastive_weight)
        self.contrastive_temperature = float(contrastive_temperature)
        self.contrastive_projection_dim = (
            int(contrastive_projection_dim) if contrastive_projection_dim is not None else None
        )
        self.contrastive_modalities = contrastive_modalities
        self.cross_modal_target_modalities = cross_modal_target_modalities
        self.min_contrastive_samples = int(min_contrastive_samples)
        self.learned_cross_modal_target = bool(learned_cross_modal_target)
        self.enable_learned_target_supervision = bool(enable_learned_target_supervision)
        self.learned_target_loss_weight = float(learned_target_loss_weight)
        self.learned_target_loss_type = str(learned_target_loss_type).lower().strip()
        self.learned_target_contrastive_weight = float(learned_target_contrastive_weight)
        self.learned_target_contrastive_temperature = (
            self.contrastive_temperature
            if learned_target_contrastive_temperature is None
            else float(learned_target_contrastive_temperature)
        )

        if self.enable_learned_target_supervision and not self.learned_cross_modal_target:
            raise ValueError(
                "enable_learned_target_supervision=True requires learned_cross_modal_target=True."
            )

        if (
            self.enable_observed_contrastive
            or self.enable_missing_contrastive
            or self.enable_learned_target_supervision
        ) and not self.perform_imputation:
            raise ValueError(
                "Contrastive/learned-target losses require impute=True "
                "so that the GCN branch is built."
            )

        if self.perform_imputation == False and self.modalities_to_impute:
            print(
                "Imputation will not be performed, so the non-empty modalities_to_impute "
                "attribute will be ignored."
            )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.use_modality_mask = use_modality_mask

        if self.perform_imputation:
            self.imputation = GCN_imputation(
                hidden_dim=d_model,
                modality_names=self.modality_names,
                graph_construction_loss_weight=self.graph_construction_loss_weight,
                stability_loss_type=self.stability_loss_type,
                mean_graph_construction_loss=self.mean_graph_construction_loss,
                gcn_refine_mode=self.gcn_refine_mode,
                modalities_to_impute=self.modalities_to_impute,
                contrastive_modalities=self.contrastive_modalities,
                enable_observed_contrastive=self.enable_observed_contrastive,
                enable_missing_contrastive=self.enable_missing_contrastive,
                observed_contrastive_weight=self.observed_contrastive_weight,
                missing_contrastive_weight=self.missing_contrastive_weight,
                contrastive_temperature=self.contrastive_temperature,
                contrastive_projection_dim=self.contrastive_projection_dim,
                cross_modal_target_modalities=self.cross_modal_target_modalities,
                min_contrastive_samples=self.min_contrastive_samples,
                learned_cross_modal_target=self.learned_cross_modal_target,
                enable_learned_target_supervision=self.enable_learned_target_supervision,
                learned_target_loss_weight=self.learned_target_loss_weight,
                learned_target_loss_type=self.learned_target_loss_type,
                learned_target_contrastive_weight=self.learned_target_contrastive_weight,
                learned_target_contrastive_temperature=self.learned_target_contrastive_temperature,
            )
        else:
            self.imputation = None

        self.fusion = MHSA_Fusion(
            hidden_dim=d_model,
            modality_names=self.modality_names,
            MHD_num_head=MHD_num_head,
            dropout=dropout,
            missingness_embedding=missingness_embedding,
            mask_missingness_fusion=mask_missingness_fusion,
        )

        # Final prediction head
        self.proj1 = nn.Linear(d_model * self.modal_num, d_model * 2)
        self.out_layer = nn.Linear(d_model * 2, output_dim)

        if self.freeze_modality_encoders:
            self.apply_frozen_modality_encoder_state()


    def _modality_encoders(self) -> dict[str, nn.Module]:
        """Return the modality-specific encoder modules."""
        return {
            modality: self.modality_encoder(modality)
            for modality in self.modality_names
        }
    
    def set_modality_encoder_trainable(
        self,
        trainable: bool = True,
    ) -> list[str]:
        """Enable or disable gradient updates for modality encoders."""
        updated: list[str] = []

        for modality, encoder in self._modality_encoders().items():
            for name, param in encoder.named_parameters():
                param.requires_grad_(trainable)
                updated.append(f"{modality}_encoder.{name}")

        return updated
    
    def apply_frozen_modality_encoder_state(self) -> list[str]:
        """
        Freeze modality encoders and the contrastive target projector, and
        keep them in evaluation mode.
        """
        if not self.freeze_modality_encoders:
            return []

        updated = self.set_modality_encoder_trainable(False)

        for encoder in self._modality_encoders().values():
            encoder.eval()

        # Freeze the CL target-side components as part of the same
        # transfer-learning state. Each component is optional, so this remains
        # safe when CL / learned targets are disabled.
        if self.imputation is not None:
            if hasattr(self.imputation, "contrastive_target_projector"):
                updated.extend(
                    self.imputation.apply_frozen_contrastive_target_projector_state()
                )
            if hasattr(self.imputation, "learned_target_networks"):
                updated.extend(
                    self.imputation.apply_frozen_learned_target_network_state()
                )

        return updated
    

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
        return self.fusion(encoded, modality_masks, return_attentions)

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
            encoded_embeds, graph_construction_loss, contrastive_losses, imputation_attentions = self.impute(
                modality_embeddings,
                modality_masks,
            )
            fused_embeds = self.fuse(encoded_embeds, modality_masks, return_aux)
            outputs = self.predict(fused_embeds["embeds"])

            if return_aux:
                # Aggregate each auxiliary objective across the modalities that
                # provide an eligible loss for that objective.
                # Per-modality losses and diagnostics remain available under their original names.
                observed_contrastive_loss = outputs.new_tensor(0.0)
                missing_contrastive_loss = outputs.new_tensor(0.0)
                observed_sample_count = outputs.new_tensor(0.0)
                missing_sample_count = outputs.new_tensor(0.0)
                observed_sample_fraction = outputs.new_tensor(0.0)
                missing_sample_fraction = outputs.new_tensor(0.0)

                observed_contrastive_modalities = 0
                missing_contrastive_modalities = 0
                learned_target_reconstruction_modalities = 0
                learned_target_contrastive_modalities = 0

                learned_target_reconstruction_loss = outputs.new_tensor(0.0)
                learned_target_contrastive_loss = outputs.new_tensor(0.0)
                learned_target_reconstruction_mean_sample_count = outputs.new_tensor(0.0)
                learned_target_reconstruction_mean_sample_fraction = outputs.new_tensor(0.0)
                learned_target_contrastive_mean_sample_count = outputs.new_tensor(0.0)
                learned_target_contrastive_mean_sample_fraction = outputs.new_tensor(0.0)

                aux_loss_components: dict[str, torch.Tensor] = {
                    "stability_loss": graph_construction_loss,
                }
                
                # Preserve per-modality losses and eligibility diagnostics for the
                # Lightning epoch-level reporting pipeline. This does not contribute
                # to the aggregate loss a second time.
                for loss_name, loss_value in contrastive_losses.items():
                    aux_loss_components[loss_name] = loss_value

                # Observed CL loss
                for modality in self.imputation.contrastive_modalities:
                    loss_key = f"observed_{modality}_contrastive_loss"
                    count_key = f"observed_{modality}_contrastive_sample_count"
                    fraction_key = f"observed_{modality}_contrastive_sample_fraction"

                    if loss_key not in contrastive_losses:
                        continue

                    count = contrastive_losses.get(count_key)

                    if count is None or int(count.item()) < self.imputation.min_contrastive_samples:
                        continue

                    observed_contrastive_loss += contrastive_losses[loss_key]
                    observed_sample_count += count
                    observed_sample_fraction += contrastive_losses[fraction_key]
                    observed_contrastive_modalities += 1
                
                # Missing CL loss
                for modality in self.imputation.contrastive_modalities:
                    loss_key = f"missing_{modality}_contrastive_loss"
                    count_key = f"missing_{modality}_contrastive_sample_count"
                    fraction_key = f"missing_{modality}_contrastive_sample_fraction"

                    if loss_key not in contrastive_losses:
                        continue

                    count = contrastive_losses.get(count_key)

                    if count is None or int(count.item()) < self.imputation.min_contrastive_samples:
                        continue

                    missing_contrastive_loss += contrastive_losses[loss_key]
                    missing_sample_count += count
                    missing_sample_fraction += contrastive_losses[fraction_key]
                    missing_contrastive_modalities += 1
                

                # Learned target reconstruction loss
                for modality in self.imputation.contrastive_modalities:
                    loss_key = f"learned_target_{modality}_reconstruction_loss"
                    count_key = f"learned_target_{modality}_reconstruction_sample_count"
                    fraction_key = f"learned_target_{modality}_reconstruction_sample_fraction"

                    if loss_key not in contrastive_losses:
                        continue

                    count = contrastive_losses.get(count_key)

                    if count is None or int(count.item()) <= 0:
                        continue

                    learned_target_reconstruction_loss += contrastive_losses[loss_key]
                    learned_target_reconstruction_mean_sample_count += count
                    learned_target_reconstruction_mean_sample_fraction += contrastive_losses[fraction_key]
                    learned_target_reconstruction_modalities += 1
                
                # Learned target InfoNCE
                for modality in self.imputation.contrastive_modalities:
                    loss_key = f"learned_target_{modality}_contrastive_loss"
                    count_key = f"learned_target_{modality}_contrastive_sample_count"
                    fraction_key = f"learned_target_{modality}_contrastive_sample_fraction"

                    if loss_key not in contrastive_losses:
                        continue

                    count = contrastive_losses.get(count_key)

                    if count is None or int(count.item()) < self.imputation.min_contrastive_samples:
                        continue

                    learned_target_contrastive_loss += contrastive_losses[loss_key]
                    learned_target_contrastive_mean_sample_count += count
                    learned_target_contrastive_mean_sample_fraction += contrastive_losses[fraction_key]
                    learned_target_contrastive_modalities += 1


                # Each auxiliary objective is averaged only across modalities that
                # contribute an eligible loss in this batch. A modality with zero
                # eligible samples must not dilute the objective with an artificial zero.
                observed_denominator = max(observed_contrastive_modalities, 1)
                missing_denominator = max(missing_contrastive_modalities, 1)
                reconstruction_denominator = max(learned_target_reconstruction_modalities, 1)
                target_contrastive_denominator = max(learned_target_contrastive_modalities, 1)

                observed_contrastive_loss = observed_contrastive_loss / observed_denominator
                missing_contrastive_loss = missing_contrastive_loss / missing_denominator
                observed_sample_count = observed_sample_count / observed_denominator
                missing_sample_count = missing_sample_count / missing_denominator
                observed_sample_fraction = observed_sample_fraction / observed_denominator
                missing_sample_fraction = missing_sample_fraction / missing_denominator
                learned_target_reconstruction_loss = learned_target_reconstruction_loss / reconstruction_denominator
                learned_target_contrastive_loss = learned_target_contrastive_loss / target_contrastive_denominator
                learned_target_reconstruction_mean_sample_count = learned_target_reconstruction_mean_sample_count / reconstruction_denominator
                learned_target_reconstruction_mean_sample_fraction = learned_target_reconstruction_mean_sample_fraction / reconstruction_denominator
                learned_target_contrastive_mean_sample_count = learned_target_contrastive_mean_sample_count / target_contrastive_denominator
                learned_target_contrastive_mean_sample_fraction = learned_target_contrastive_mean_sample_fraction / target_contrastive_denominator

                weighted_observed_contrastive_loss = (
                    self.imputation.observed_contrastive_weight * observed_contrastive_loss
                )
                weighted_missing_contrastive_loss = (
                    self.imputation.missing_contrastive_weight * missing_contrastive_loss
                )
                weighted_contrastive_loss = (
                    weighted_observed_contrastive_loss + weighted_missing_contrastive_loss
                )
                weighted_learned_target_reconstruction_loss = (
                    self.imputation.learned_target_loss_weight * learned_target_reconstruction_loss
                )
                weighted_learned_target_contrastive_loss = (
                    self.imputation.learned_target_contrastive_weight * learned_target_contrastive_loss
                )
                weighted_learned_target_loss = (
                    weighted_learned_target_reconstruction_loss
                    + weighted_learned_target_contrastive_loss
                )

                # Always expose stable aggregate names, even when a branch is disabled.
                aux_loss_components["learned_target_reconstruction_loss"] = learned_target_reconstruction_loss
                aux_loss_components["learned_target_contrastive_loss"] = learned_target_contrastive_loss
                aux_loss_components["weighted_learned_target_reconstruction_loss"] = weighted_learned_target_reconstruction_loss
                aux_loss_components["weighted_learned_target_contrastive_loss"] = weighted_learned_target_contrastive_loss
                aux_loss_components["weighted_learned_target_loss"] = weighted_learned_target_loss
                aux_loss_components["learned_target_reconstruction_mean_sample_count"] = learned_target_reconstruction_mean_sample_count
                aux_loss_components["learned_target_reconstruction_mean_sample_fraction"] = learned_target_reconstruction_mean_sample_fraction
                aux_loss_components["learned_target_contrastive_mean_sample_count"] = learned_target_contrastive_mean_sample_count
                aux_loss_components["learned_target_contrastive_mean_sample_fraction"] = learned_target_contrastive_mean_sample_fraction
                
                aux_loss_components["observed_contrastive_loss"] = observed_contrastive_loss
                aux_loss_components["missing_contrastive_loss"] = missing_contrastive_loss
                aux_loss_components["weighted_observed_contrastive_loss"] = weighted_observed_contrastive_loss
                aux_loss_components["weighted_missing_contrastive_loss"] = weighted_missing_contrastive_loss
                aux_loss_components["weighted_contrastive_loss"] = weighted_contrastive_loss

                # Diagnostics that disambiguate a disabled loss from an enabled loss
                # with too few eligible samples.
                aux_loss_components["observed_contrastive_enabled"] = outputs.new_tensor(
                    float(self.imputation.enable_observed_contrastive)
                )
                aux_loss_components["missing_contrastive_enabled"] = outputs.new_tensor(
                    float(self.imputation.enable_missing_contrastive)
                )
                aux_loss_components["observed_contrastive_mean_sample_count"] = observed_sample_count
                aux_loss_components["missing_contrastive_mean_sample_count"] = missing_sample_count
                aux_loss_components["observed_contrastive_mean_sample_fraction"] = observed_sample_fraction
                aux_loss_components["missing_contrastive_mean_sample_fraction"] = missing_sample_fraction

                # Learned-target supervision diagnostics.
                learned_target_supervision_enabled = outputs.new_tensor(
                    float(self.imputation.enable_learned_target_supervision)
                )
                aux_loss_components["learned_target_supervision_enabled"] = learned_target_supervision_enabled

                aux_loss_components["total_aux_loss"] = (
                    graph_construction_loss
                    + weighted_contrastive_loss
                    + weighted_learned_target_loss
                )

                outputs_dict = {
                    "logits": outputs,
                    "aux_loss": (
                        graph_construction_loss
                        + weighted_contrastive_loss
                        + weighted_learned_target_loss
                    ),
                    "aux_loss_components": aux_loss_components,
                    "imputation_attentions": imputation_attentions,
                    "fusion_attentions": fused_embeds["fusion_attentions"],
                }

        else:
            fused_embeds = self.fuse(modality_embeddings, modality_masks, return_aux)
            outputs = self.predict(fused_embeds["embeds"])

            if return_aux:
                outputs_dict = {
                    "logits": outputs,
                    "fusion_attentions": fused_embeds["fusion_attentions"],
                }

        if return_aux:
            return outputs_dict
        else:
            return outputs

    def _modality_mask(
        self,
        batch: Mapping[str, torch.Tensor],
        encoded: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not self.use_modality_mask:
            first = encoded[self.modality_names[0]]
            mask = first.new_ones(first.shape[0], 1).float()
            return {modality: mask for modality in self.modality_names}

        device = encoded[self.modality_names[0]].device
        return {
            modality: to_binary_mask(modality_available(batch, modality, device=device))
            for modality in self.modality_names
        }


def build_m3care_model_from_config(
    fusion_cfg: Mapping[str, Any],
    feature_dims: Mapping[str, int],
    output_dim: int = 1,
) -> M3CareModel:
    """Build a baseline M3Care model from fusion config and dataset feature dimensions."""
    print(f"Dimensions of modality embeddings: {feature_dims}")
    modalities_to_impute = fusion_cfg.get("modalities_to_impute", None)
    if modalities_to_impute is not None:
        modalities_to_impute = list(modalities_to_impute)

    contrastive_modalities = fusion_cfg.get("contrastive_modalities", None)
    if contrastive_modalities is not None:
        contrastive_modalities = list(contrastive_modalities)

    cross_modal_target_modalities = fusion_cfg.get("cross_modal_target_modalities", None)
    if cross_modal_target_modalities is not None:
        cross_modal_target_modalities = list(cross_modal_target_modalities)

    enable_observed_contrastive = fusion_cfg.get("enable_observed_contrastive", False)
    enable_missing_contrastive = fusion_cfg.get("enable_missing_contrastive", False)
    observed_contrastive_weight = fusion_cfg.get("observed_contrastive_weight", 1.0)
    missing_contrastive_weight = fusion_cfg.get("missing_contrastive_weight", 1.0)
    learned_cross_modal_target = fusion_cfg.get("learned_cross_modal_target", False)
    enable_learned_target_supervision = fusion_cfg.get("enable_learned_target_supervision", False)
    learned_target_loss_weight = fusion_cfg.get("learned_target_loss_weight", 1.0)
    learned_target_loss_type = fusion_cfg.get("learned_target_loss_type", "cosine")
    learned_target_contrastive_weight = fusion_cfg.get("learned_target_contrastive_weight", 1.0)
    learned_target_contrastive_temperature = fusion_cfg.get("learned_target_contrastive_temperature", None)

    return M3CareModel(
        d_model=int(fusion_cfg.get("d_model", 128)),
        modality_names=list(fusion_cfg.get("modality_names", feature_dims.keys())),
        output_dim=int(fusion_cfg.get("output_dim", output_dim)),
        dropout=float(fusion_cfg.get("dropout", 0.0)),
        MHD_num_head = int(fusion_cfg.get("MHD_num_head", 4)),
        use_modality_mask=bool(fusion_cfg.get("use_modality_mask", True)),
        impute=bool(fusion_cfg.get("impute", False)),
        missingness_embedding=bool(fusion_cfg.get("missingness_embedding", False)),
        graph_construction_loss_weight=float(fusion_cfg.get("graph_construction_loss_weight", 0.0)),
        stability_loss_type=str(fusion_cfg.get("stability_loss_type", "norm")),
        mean_graph_construction_loss=bool(fusion_cfg.get("mean_graph_construction_loss", False)),
        gcn_refine_mode=str(fusion_cfg.get("gcn_refine_mode", "adaptive")),
        mask_missingness_fusion=bool(fusion_cfg.get("mask_missingness_fusion", False)),
        modalities_to_impute=modalities_to_impute,
        feature_dims=dict(feature_dims),
        contrastive_modalities=contrastive_modalities,
        enable_observed_contrastive=bool(enable_observed_contrastive),
        enable_missing_contrastive=bool(enable_missing_contrastive),
        observed_contrastive_weight=float(observed_contrastive_weight),
        missing_contrastive_weight=float(missing_contrastive_weight),
        contrastive_temperature=float(fusion_cfg.get("contrastive_temperature", 0.1)),
        contrastive_projection_dim=fusion_cfg.get("contrastive_projection_dim", None),
        cross_modal_target_modalities=cross_modal_target_modalities,
        min_contrastive_samples=int(fusion_cfg.get("min_contrastive_samples", 2)),
        learned_cross_modal_target=bool(learned_cross_modal_target),
        enable_learned_target_supervision=bool(enable_learned_target_supervision),
        learned_target_loss_weight=float(learned_target_loss_weight),
        learned_target_loss_type=str(learned_target_loss_type),
        learned_target_contrastive_weight=float(learned_target_contrastive_weight),
        learned_target_contrastive_temperature=(
            None if learned_target_contrastive_temperature is None
            else float(learned_target_contrastive_temperature)
        ),
        freeze_modality_encoders=bool(fusion_cfg.get("freeze_modality_encoders", False)),
    )



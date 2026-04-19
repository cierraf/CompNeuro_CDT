from typing import List, Dict, Sequence, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================
# Encoder
# ==============================
class ModalityEncoder(nn.Module):
    """
    Input:  x_mod [B,T,D_in]
    Output: z_mod [B,T_out,D_model]
    """
    def __init__(
        self,
        d_in: int,
        d_hidden: int = 128,
        d_model: int = 64,
        dilation: int = 1,
        stride: int = 1,
        b_use_maxpooling_layer: bool = False,
        k: int = 7,
        n_temporal_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        pad = dilation * (k // 2)

        self.stride = stride
        self.k = k
        self.pad = pad
        self.n_temporal_layers = n_temporal_layers
        self.b_use_maxpooling_layer = b_use_maxpooling_layer

        # feature projection
        self.feat_proj = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # temporal conv stack
        layers = []
        for _ in range(n_temporal_layers):
            layers += [
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size=k,
                    dilation=dilation,
                    stride=stride,
                    padding=pad,
                ),
                nn.ReLU(),
            ]

            if self.b_use_maxpooling_layer:
                layers.append(
                    nn.MaxPool1d(
                        kernel_size=k,
                        stride=stride,
                        padding=pad,
                    )
                )

            layers.append(nn.Dropout(dropout))

        self.temp_block = nn.Sequential(*layers)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_mod: torch.Tensor):
        z = self.feat_proj(x_mod)   # [B, T, d_model]
        y = z.transpose(1, 2)       # [B, d_model, T]
        y = self.temp_block(y)      # [B, d_model, T_out]

        z = y.transpose(1, 2)       # [B, T_out, d_model]
        z = self.norm(z)

        return z

# ==============================
# Time Sequence
# ==============================

class SequentialModel(nn.Module):
    """
    Unified sequential encoder for comparison among:
        - GRU
        - LSTM
        - Transformer

    Input:
        x:    [B, T, D]

    Output:
        h_seq: [B, T, D_seq]
        h:     [B, D_out]
    """
    def __init__(
        self,
        d_in: int,
        model_type: str = "gru",
        d_hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        nhead: int = 4,
        dim_feedforward: int = 256,
        out_pool: str = "mean",
        last_window_size: int = 3,
        last_window_mode: int = "max",
        **kwargs
    ):
        super().__init__()

        model_type = model_type.lower()
        out_pool = out_pool.lower()
        last_window_mode = last_window_mode.lower()

        assert model_type in {"gru", "lstm", "transformer"}
        assert out_pool in {"last", "mean", "last_window", "attn", "max"}
        assert last_window_mode in {"mean", "max"}
        assert last_window_size >= 1

        self.model_type = model_type
        self.out_pool = out_pool
        self.bidirectional = bidirectional
        self.last_window_size = last_window_size
        self.last_window_mode = last_window_mode

        # -----------------------------
        # sequential encoder
        # -----------------------------
        if model_type in {"gru", "lstm"}:
            rnn_cls = nn.GRU if model_type == "gru" else nn.LSTM

            self.seq = rnn_cls(
                input_size=d_in,
                hidden_size=d_hidden,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )

            self.seq_d_out = d_hidden * (2 if bidirectional else 1)

        elif model_type == "transformer":
            assert d_in % nhead == 0, "For transformer, d_in must be divisible by nhead."

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_in,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=False,
            )
            self.seq = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
            self.seq_d_out = d_in

        # -----------------------------
        # attention pooling head
        # -----------------------------
        if self.out_pool == "attn":
            self.attn_score = nn.Linear(self.seq_d_out, 1)

        self.d_out = self.seq_d_out

    @staticmethod
    def mean_pool(x, mask=None):
        if mask is None:
            return x.mean(dim=1)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)

    @staticmethod
    def max_pool(x, mask=None):
        if mask is None:
            return x.max(dim=1).values
        x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        return x.max(dim=1).values

    @staticmethod
    def last_pool(x, mask=None):
        if mask is None:
            return x[:, -1, :]
        idx = mask.sum(dim=1) - 1
        return x[torch.arange(x.size(0), device=x.device), idx]

    @staticmethod
    def last_window_pool(x: torch.Tensor, window: int = 5, mode: str = "mean") -> torch.Tensor:
        B, T, D = x.shape
        outputs = []

        for b in range(B):
            segment = x[b, max(0, T - window):T]

            if mode == "mean":
                pooled = segment.mean(dim=0)
            elif mode == "max":
                pooled = segment.max(dim=0).values
            else:
                raise ValueError("mode must be 'mean' or 'max'")

            outputs.append(pooled)

        return torch.stack(outputs, dim=0)

    def attn_pool(self, x: torch.Tensor):
        """
        x: [B, T, D]

        returns:
            pooled: [B, D]
            attn_w: [B, T]
        """
        logits = self.attn_score(x).squeeze(-1)   # [B, T]
        attn_w = torch.softmax(logits, dim=1)     # [B, T]
        pooled = torch.sum(x * attn_w.unsqueeze(-1), dim=1)  # [B, D]
        return pooled, attn_w
    
    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        x:    [B, T, D_in]
        mask: [B, T] (1=valid, 0=pad)
        """

        # -----------------------------
        # sequential encoding
        # -----------------------------
        if self.model_type in {"gru", "lstm"}:
            h_seq, _ = self.seq(x)
        elif self.model_type == "transformer":
            h_seq = self.seq(
                x
            )

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # -----------------------------
        # global pooling
        # -----------------------------
        attn_w = None

        if self.out_pool == "last":
            h_global = self.last_pool(h_seq)

        elif self.out_pool == "mean":
            h_global = self.mean_pool(h_seq)

        elif self.out_pool == "max":
            h_global = self.max_pool(h_seq)

        elif self.out_pool == "last_window":
            h_global = self.last_window_pool(
                h_seq,
                window=self.last_window_size,
                mode=self.last_window_mode,
            )

        elif self.out_pool == "attn":
            h_global, attn_w = self.attn_pool(h_seq)

        else:
            raise ValueError(f"Unknown out_pool: {self.out_pool}")

        h = h_global

        if return_attn and self.out_pool == "attn":
            return h_seq, h, attn_w

        return h_seq, h
    
# ==============================
# Outhead
# ==============================
class FullyConnectedHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.out = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = True,
        return_probs: bool = False,
        return_pred: bool = False,
        threshold: float = 0.5,
    ):
        logits = self.out(x)   # [B, 3]

        probs = None
        pred = None

        if return_probs or return_pred:
            probs = torch.sigmoid(logits)

        if return_pred:
            pred = (probs >= threshold).long()

        if return_logits and not return_probs and not return_pred:
            return logits

        out = {}
        if return_logits:
            out["logits"] = logits
        if return_probs:
            out["probs"] = probs
        if return_pred:
            out["pred"] = pred

        return out

class EyeDecoderHead(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=128,
        out_gaze_dim=2,
        out_pupil_dim=1,
        dropout=0.1,
        use_head_dropout=True,
    ):
        super().__init__()

        # shared trunk
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # gaze head
        gaze_layers = [
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
        ]
        if use_head_dropout:
            gaze_layers.append(nn.Dropout(dropout))

        gaze_layers.append(nn.Linear(32, out_gaze_dim))
        self.gaze_head = nn.Sequential(*gaze_layers)

        # pupil head
        pupil_layers = [
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
        ]
        if use_head_dropout:
            pupil_layers.append(nn.Dropout(dropout))

        pupil_layers.append(nn.Linear(32, out_pupil_dim))
        self.pupil_head = nn.Sequential(*pupil_layers)

    def forward(self, h):
        h = self.shared(h)

        gaze_hat = self.gaze_head(h)
        pupil_hat = self.pupil_head(h)

        return torch.cat([gaze_hat, pupil_hat], dim=-1)
    
class FaceLabelDecoderHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = True,
        return_probs: bool = False,
        return_pred: bool = False,
        threshold: float = 0.5,
    ):
        """
        x: [B, in_dim]

        returns:
            - logits: [B, 3]
            - probs:  [B, 3]
            - pred:   [B, 3]  (0/1)
        """
        logits = self.net(x)   # [B, 3]

        probs = None
        pred = None

        if return_probs or return_pred:
            probs = torch.sigmoid(logits)

        if return_pred:
            pred = (probs >= threshold).long()

        if return_logits and not return_probs and not return_pred:
            return logits

        out = {}
        if return_logits:
            out["logits"] = logits
        if return_probs:
            out["probs"] = probs
        if return_pred:
            out["pred"] = pred

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F

from new_code.neural_network import model_modules


class MultiModalFusionModel(nn.Module):
    def __init__(
        self,
        modalities,   # list[str] or dict
        n_temporal_layers: int = 3,
        b_mod_early_fusion: bool = True,
        d_encoder_dim: int = 32,
        d_encoder_hidden: int = 128,
        encoder_time_dilation: int = 1,
        encoder_time_kernel: int = 7,
        b_use_maxpooling_layer: bool = True,
        sequential_model_hidden: int = 128,
        sequential_model_type: str = "lstm",
        sequential_model_layers: int = 1,
        sequential_pooling: str = "attn",
        dropout: float = 0.1,
        outhead: str = "fcl",
        label_dim: int = 3,
        **kwargs
    ):
        super().__init__()

        supported_modalities = {
            "firing_rate",
            "lfp_micro",
            "lfp_macro",
            "eye_gaze",
            "pupil",
        }

        # -----------------------------
        # normalize modalities input
        # -----------------------------
        if isinstance(modalities, (list, tuple, set)):
            self.modalities = list(modalities)
            self.modality_cfg = {m: {} for m in self.modalities}
        elif isinstance(modalities, dict):
            self.modalities = list(modalities.keys())
            self.modality_cfg = modalities
        else:
            raise TypeError("`modalities` must be list[str] or dict")

        unsupported = set(self.modalities) - supported_modalities
        if len(unsupported) > 0:
            raise ValueError(f"Unsupported modalities: {unsupported}")

        last_window_mode = kwargs.get("last_window_mode", "mean")
        last_window_size = kwargs.get("last_window_size", 5)

        # -----------------------------
        # modality dims
        # if not provided, use lazy infer in first forward
        # -----------------------------
        self.d_firing = self.modality_cfg.get("firing_rate", {}).get("dim", None)
        self.d_lfp_micro = self.modality_cfg.get("lfp_micro", {}).get("dim", None)
        self.d_lfp_macro = self.modality_cfg.get("lfp_macro", {}).get("dim", None)
        self.d_gaze = self.modality_cfg.get("eye_gaze", {}).get("dim", None)
        self.d_pupil = self.modality_cfg.get("pupil", {}).get("dim", None)

        self.b_mod_early_fusion = b_mod_early_fusion
        self.d_encoder_dim = d_encoder_dim
        self.d_encoder_hidden = d_encoder_hidden
        self.encoder_time_dilation = encoder_time_dilation
        self.encoder_time_kernel = encoder_time_kernel
        self.b_use_maxpooling_layer = b_use_maxpooling_layer
        self.n_temporal_layers = n_temporal_layers
        self.dropout = dropout

        # encoders
        self.all_mods_enc = None
        self.firing_rate_enc = None
        self.lfp_macro_enc = None
        self.lfp_micro_enc = None
        self.gaze_enc = None
        self.pupil_enc = None

        # 先不完全依赖 dim，支持 lazy build
        if self._all_dims_known():
            self._build_encoders()

        # -----------------------------
        # temporal model
        # -----------------------------
        if b_mod_early_fusion:
            d_seq_in = d_encoder_dim
        else:
            d_seq_in = d_encoder_dim * len(self.modalities)

        self.sequential_model = model_modules.SequentialModel(
            d_in=d_seq_in,
            d_hidden=sequential_model_hidden,
            model_type=sequential_model_type,
            num_layers=sequential_model_layers,
            dropout=dropout,
            out_pool=sequential_pooling,
            last_window_size=last_window_size,
            last_window_mode=last_window_mode,
        )
        fused_out_dim = self.sequential_model.d_out

        # -----------------------------
        # output head
        # -----------------------------
        self.face_decode_mode = None
        if outhead == "fcl":
            self.out = model_modules.FullyConnectedHead(
                fused_out_dim, label_dim, dropout=dropout
            )
        elif outhead == "face_decode":
            self.face_decode_mode = kwargs.get("face_decode_mode", "logits")
            self.out = model_modules.FaceLabelDecoderHead(
                fused_out_dim, out_dim=label_dim, dropout=dropout
            )
        elif outhead == "eye_decode":
            self.out = model_modules.EyeDecoderHead(
                fused_out_dim, dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported outhead: {outhead}")

    def _all_dims_known(self):
        dim_map = {
            "firing_rate": self.d_firing,
            "lfp_micro": self.d_lfp_micro,
            "lfp_macro": self.d_lfp_macro,
            "eye_gaze": self.d_gaze,
            "pupil": self.d_pupil,
        }
        for m in self.modalities:
            if dim_map[m] is None:
                return False
        return True

    def _infer_dims_from_x(self, x: dict):
        if "firing_rate" in x and self.d_firing is None:
            self.d_firing = x["firing_rate"].shape[-1]
        if "lfp_micro" in x and self.d_lfp_micro is None:
            self.d_lfp_micro = x["lfp_micro"].shape[-1]
        if "lfp_macro" in x and self.d_lfp_macro is None:
            self.d_lfp_macro = x["lfp_macro"].shape[-1]
        if "eye_gaze" in x and self.d_gaze is None:
            self.d_gaze = x["eye_gaze"].shape[-1]
        if "pupil" in x and self.d_pupil is None:
            self.d_pupil = x["pupil"].shape[-1]

    def _build_encoders(self):
        d_all_mods = 0
        if "firing_rate" in self.modalities:
            d_all_mods += self.d_firing
        if "lfp_micro" in self.modalities:
            d_all_mods += self.d_lfp_micro
        if "lfp_macro" in self.modalities:
            d_all_mods += self.d_lfp_macro
        if "eye_gaze" in self.modalities:
            d_all_mods += self.d_gaze
        if "pupil" in self.modalities:
            d_all_mods += self.d_pupil

        if self.b_mod_early_fusion:
            self.all_mods_enc = model_modules.ModalityEncoder(
                d_in=d_all_mods,
                d_hidden=self.d_encoder_hidden,
                d_model=self.d_encoder_dim,
                dilation=self.encoder_time_dilation,
                k=self.encoder_time_kernel,
                b_use_maxpooling_layer=self.b_use_maxpooling_layer,
                n_temporal_layers=self.n_temporal_layers,
                dropout=self.dropout,
            )
        else:
            self.firing_rate_enc = self.build_encoder(self.d_firing)
            self.lfp_micro_enc = self.build_encoder(self.d_lfp_micro)
            self.lfp_macro_enc = self.build_encoder(self.d_lfp_macro)
            self.gaze_enc = self.build_encoder(self.d_gaze)
            self.pupil_enc = self.build_encoder(self.d_pupil)

    def forward(self, x: dict):
        if not isinstance(x, dict):
            raise TypeError("Model input `x` must be a dict of modality tensors")

        # lazy infer + build
        if self.all_mods_enc is None and self.firing_rate_enc is None and self.lfp_micro_enc is None \
           and self.lfp_macro_enc is None and self.gaze_enc is None and self.pupil_enc is None:
            self._infer_dims_from_x(x)
            self._build_encoders()

        x_list = []

        x_firing = x.get("firing_rate", None)
        x_lfp_micro = x.get("lfp_micro", None)
        x_lfp_macro = x.get("lfp_macro", None)
        x_gaze = x.get("eye_gaze", None)
        x_pupil = x.get("pupil", None)

        if self.all_mods_enc is not None:
            if "firing_rate" in self.modalities:
                x_list.append(x_firing)
            if "lfp_micro" in self.modalities:
                x_list.append(x_lfp_micro)
            if "lfp_macro" in self.modalities:
                x_list.append(x_lfp_macro)
            if "eye_gaze" in self.modalities:
                x_list.append(x_gaze)
            if "pupil" in self.modalities:
                x_list.append(x_pupil)

            if any(xx is None for xx in x_list):
                missing = [m for m in self.modalities if x.get(m, None) is None]
                raise ValueError(f"Missing modalities in batch: {missing}")

            x_all_mods = torch.cat(x_list, dim=-1)
            z_all_mods = self.all_mods_enc(x_all_mods)

        else:
            z_list = []

            if "firing_rate" in self.modalities:
                z_list.append(self.firing_rate_enc(x_firing))
            if "lfp_micro" in self.modalities:
                z_list.append(self.lfp_micro_enc(x_lfp_micro))
            if "lfp_macro" in self.modalities:
                z_list.append(self.lfp_macro_enc(x_lfp_macro))
            if "eye_gaze" in self.modalities:
                z_list.append(self.gaze_enc(x_gaze))
            if "pupil" in self.modalities:
                z_list.append(self.pupil_enc(x_pupil))

            z_all_mods = torch.cat(z_list, dim=-1)

        z_seq, z_pooled = self.sequential_model(z_all_mods)

        y_hat = self.out(z_pooled)

        return z_seq, y_hat

    def build_encoder(self, d_in):
        if d_in is None or d_in == 0:
            return None

        return model_modules.ModalityEncoder(
            d_in=d_in,
            d_hidden=self.d_encoder_hidden,
            d_model=self.d_encoder_dim,
            dilation=self.encoder_time_dilation,
            k=self.encoder_time_kernel,
            b_use_maxpooling_layer=self.b_use_maxpooling_layer,
            n_temporal_layers=self.n_temporal_layers,
            dropout=self.dropout,
        )
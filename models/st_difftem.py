"""ST-DiffTEM diffusion variants built on frozen TEM baselines."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .single_receiver_pcbilstm import IndependentReceiverPCBiLSTM
from .spatial_receiver_pcbilstm import GraphBiasedReceiverAttention, SpatialReceiverPCBiLSTM


VALID_VARIANTS = ("spatial_base", "hybrid_base")
VALID_PREDICTION_TYPES = ("noise", "x0")
VALID_RESIDUAL_INFERENCE_MODES = ("ddim_random", "anchor")
VALID_RESIDUAL_ANCHOR_MODES = ("zero", "observed")


def _make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Return a GroupNorm layer whose group count divides num_channels."""
    for groups in (max_groups, 4, 2, 1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_checkpoint_payload(checkpoint_path: Path) -> Any:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("Unsupported checkpoint payload.")


class SinusoidalStepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, step_idx: torch.Tensor) -> torch.Tensor:
        if step_idx.ndim != 1:
            raise ValueError(f"Expected 1D step_idx, got {tuple(step_idx.shape)}")
        device = step_idx.device
        half_dim = max(self.dim // 2, 1)
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        freq = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -scale)
        angles = step_idx.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class FrozenSpatialAdapter(nn.Module):
    """Frozen receiver-aware denoiser used as a deterministic base."""

    def __init__(self, spatial_model: nn.Module):
        super().__init__()
        self.model = spatial_model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @classmethod
    def from_artifact_dir(cls, artifact_dir: Path) -> tuple["FrozenSpatialAdapter", dict[str, Any]]:
        run_config_path = artifact_dir / "run_config.json"
        checkpoint_path = artifact_dir / "best_model.pth"
        if not run_config_path.exists():
            raise FileNotFoundError(f"Missing Spatial run_config: {run_config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing Spatial checkpoint: {checkpoint_path}")

        run_config = load_json(run_config_path)
        model_cfg = dict(run_config.get("model", {}))
        if run_config.get("model_type") == "external_baseline":
            model_name = run_config.get("model_name")
            if model_name != "dncnn2d":
                raise ValueError(f"Unsupported heterogeneous spatial expert: {model_name}")
            from .external_baselines import build_external_baseline

            spatial_model = build_external_baseline(model_name, model_cfg)
        else:
            spatial_model = SpatialReceiverPCBiLSTM(
                seq_len=int(model_cfg.get("seq_len", 869)),
                num_receivers=int(model_cfg.get("num_receivers", 11)),
                filters=int(model_cfg.get("filters", 16)),
                lstm_hidden=int(model_cfg.get("lstm_hidden", 32)),
                lstm_layers=int(model_cfg.get("lstm_layers", 2)),
                num_heads=int(model_cfg.get("num_heads", 4)),
                spatial_layers=int(model_cfg.get("spatial_layers", 2)),
                dropout=float(model_cfg.get("dropout", 0.1)),
                use_graph_bias=bool(model_cfg.get("use_graph_bias", True)),
                residual=bool(model_cfg.get("residual", True)),
            )
        payload = _load_checkpoint_payload(checkpoint_path)
        spatial_model.load_state_dict(_extract_state_dict(payload))
        return cls(spatial_model), run_config

    def forward(self, noisy_log: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(noisy_log)


class FrozenSingleAdapter(nn.Module):
    """Frozen independent single-receiver baseline used as an auxiliary base."""

    def __init__(self, single_model: IndependentReceiverPCBiLSTM):
        super().__init__()
        self.model = single_model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @classmethod
    def from_artifact_dir(cls, artifact_dir: Path) -> tuple["FrozenSingleAdapter", dict[str, Any]]:
        run_config_path = artifact_dir / "run_config.json"
        checkpoint_path = artifact_dir / "best_model.pth"
        if not run_config_path.exists():
            raise FileNotFoundError(f"Missing Single run_config: {run_config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing Single checkpoint: {checkpoint_path}")

        run_config = load_json(run_config_path)
        model_cfg = dict(run_config.get("model", {}))
        single_model = IndependentReceiverPCBiLSTM(
            seq_len=int(model_cfg.get("seq_len", 869)),
            num_receivers=int(model_cfg.get("num_receivers", 11)),
            filters=int(model_cfg.get("filters", 16)),
            lstm_hidden=int(model_cfg.get("lstm_hidden", 32)),
            lstm_layers=int(model_cfg.get("lstm_layers", 2)),
            num_heads=int(model_cfg.get("num_heads", 4)),
        )
        payload = _load_checkpoint_payload(checkpoint_path)
        single_model.load_state_dict(_extract_state_dict(payload))
        return cls(single_model), run_config

    def forward(self, noisy_log: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(noisy_log)


class ConditionEncoder(nn.Module):
    """Local temporal encoder plus receiver-aware spatial mixing."""

    def __init__(
        self,
        in_channels: int,
        num_receivers: int = 11,
        hidden_channels: int = 32,
        feature_dim: int = 64,
        spatial_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_receivers = int(num_receivers)
        self.hidden_channels = int(hidden_channels)
        self.feature_dim = int(feature_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)

        self.stem = nn.Sequential(
            nn.Conv1d(self.in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, feature_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(feature_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.SiLU(),
        )
        self.receiver_embedding = nn.Parameter(torch.zeros(1, 1, self.num_receivers, feature_dim))
        nn.init.normal_(self.receiver_embedding, mean=0.0, std=0.02)
        self.spatial_blocks = nn.ModuleList(
            [
                GraphBiasedReceiverAttention(
                    dim=feature_dim,
                    num_receivers=self.num_receivers,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_graph_bias=True,
                    learnable_graph_scale=True,
                )
                for _ in range(spatial_layers)
            ]
        )

    def forward(self, stack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # stack: [B, R, C, T]
        bsz, num_receivers, in_channels, seq_len = stack.shape
        if num_receivers != self.num_receivers:
            raise ValueError(f"Expected {self.num_receivers} receivers, got {num_receivers}")
        if in_channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} condition channels, got {in_channels}")

        local = self.stem(stack.reshape(bsz * num_receivers, in_channels, seq_len))
        local = local.reshape(bsz, num_receivers, self.feature_dim, seq_len)
        spatial = local.permute(0, 3, 1, 2).contiguous() + self.receiver_embedding
        for block in self.spatial_blocks:
            spatial = block(spatial)
        spatial = spatial.permute(0, 2, 3, 1).contiguous()  # [B, R, D, T]
        return local, spatial


class ReceiverGateNet(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_receivers: int = 11,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        init_bias: float = 0.0,
        zero_init_output: bool = False,
    ):
        super().__init__()
        self.num_receivers = int(num_receivers)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.init_bias = float(init_bias)
        self.zero_init_output = bool(zero_init_output)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        output_layer = self.mlp[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("ReceiverGateNet expects the final MLP layer to be nn.Linear.")
        if self.zero_init_output:
            nn.init.zeros_(output_layer.weight)
        nn.init.constant_(output_layer.bias, self.init_bias)

    def forward(self, local_feat: torch.Tensor, spatial_feat: torch.Tensor) -> torch.Tensor:
        bsz, num_receivers, _, _ = local_feat.shape
        if num_receivers != self.num_receivers:
            raise ValueError(f"Expected {self.num_receivers} receivers, got {num_receivers}")
        local_summary = self.pool(local_feat.reshape(bsz * num_receivers, local_feat.shape[2], local_feat.shape[3])).squeeze(-1)
        spatial_summary = self.pool(spatial_feat.reshape(bsz * num_receivers, spatial_feat.shape[2], spatial_feat.shape[3])).squeeze(-1)
        summary = torch.cat([local_summary, spatial_summary], dim=1)
        gate = torch.sigmoid(self.mlp(summary)).reshape(bsz, num_receivers, 1)
        return gate


class TemporalResidualGateNet(nn.Module):
    """Time-resolved residual value gate, initialized to be conservative.

    The receiver-level gate decides whether a receiver should be corrected at
    all. This module adds a temporal mask so dropout segments and local corruptions
    can be corrected without opening the whole 869-sample trace.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_channels: int = 64,
        dropout: float = 0.1,
        init_bias: float = 0.0,
        zero_init_output: bool = True,
    ):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.dropout = float(dropout)
        self.init_bias = float(init_bias)
        self.zero_init_output = bool(zero_init_output)
        input_channels = int(feature_dim) * 2 + 3
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, self.hidden_channels, kernel_size=3, padding=1),
            _make_group_norm(self.hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=5, padding=2),
            _make_group_norm(self.hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(self.hidden_channels, 1, kernel_size=3, padding=1),
        )
        output_layer = self.net[-1]
        if not isinstance(output_layer, nn.Conv1d):
            raise TypeError("TemporalResidualGateNet expects the final layer to be nn.Conv1d.")
        if self.zero_init_output:
            nn.init.zeros_(output_layer.weight)
        nn.init.constant_(output_layer.bias, self.init_bias)

    def forward(
        self,
        local_feat: torch.Tensor,
        spatial_feat: torch.Tensor,
        noisy_log: torch.Tensor,
        base_log: torch.Tensor,
    ) -> torch.Tensor:
        bsz, num_receivers, _, seq_len = local_feat.shape
        observed_residual = noisy_log - base_log
        x = torch.cat(
            [
                local_feat,
                spatial_feat,
                noisy_log.unsqueeze(2),
                base_log.unsqueeze(2),
                observed_residual.abs().unsqueeze(2),
            ],
            dim=2,
        )
        x = x.reshape(bsz * num_receivers, x.shape[2], seq_len)
        gate = torch.sigmoid(self.net(x)).reshape(bsz, num_receivers, seq_len)
        return gate


class ResidualDiffusionHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_channels: int = 128,
        step_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.step_dim = int(step_dim)
        self.dropout = float(dropout)
        self.step_embed = SinusoidalStepEmbedding(step_dim)
        self.step_mlp = nn.Sequential(
            nn.Linear(step_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        input_channels = feature_dim * 2 + 4
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.norm1 = _make_group_norm(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.norm2 = _make_group_norm(hidden_channels)
        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.norm3 = _make_group_norm(hidden_channels)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        local_feat: torch.Tensor,
        spatial_feat: torch.Tensor,
        noisy_log: torch.Tensor,
        base_log: torch.Tensor,
        residual_state: torch.Tensor,
        step_idx: torch.Tensor,
    ) -> torch.Tensor:
        bsz, num_receivers, _, seq_len = local_feat.shape
        observed_residual = noisy_log - base_log
        x = torch.cat(
            [
                local_feat,
                spatial_feat,
                noisy_log.unsqueeze(2),
                base_log.unsqueeze(2),
                observed_residual.unsqueeze(2),
                residual_state.unsqueeze(2),
            ],
            dim=2,
        )
        x = x.reshape(bsz * num_receivers, x.shape[2], seq_len)
        step_emb = self.step_mlp(self.step_embed(step_idx)).unsqueeze(-1)
        step_emb = step_emb.repeat_interleave(num_receivers, dim=0)
        h = self.conv1(x) + step_emb
        h = F.silu(self.norm1(h), inplace=True)
        residual = h
        h = F.silu(self.norm2(self.conv2(h)), inplace=True)
        h = self.drop(h)
        h = F.silu(self.norm3(self.conv3(h) + residual), inplace=True)
        h = self.drop(h)
        return self.out(h).reshape(bsz, num_receivers, seq_len)


class STDiffTEMNet(nn.Module):
    def __init__(
        self,
        seq_len: int = 869,
        num_receivers: int = 11,
        variant: str = "spatial_base",
        condition_feature_dim: int = 64,
        condition_hidden_channels: int = 32,
        condition_spatial_layers: int = 1,
        gate_hidden_dim: int = 64,
        diffusion_hidden_channels: int = 128,
        diffusion_steps: int = 16,
        sample_steps: int = 8,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        step_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        refine_scale: float = 1.0,
        sampler_seed: int = 20260430,
        prediction_type: str = "x0",
        residual_inference_mode: str = "anchor",
        residual_anchor_mode: str = "zero",
        residual_clip_value: float = 0.5,
        max_residual_gate: float = 0.5,
        rho_max: float = 0.5,
        residual_scale_by_w_single: bool = True,
        residual_scale_floor: float = 0.5,
        use_temporal_gate: bool = True,
        temporal_gate_init_bias: float = 0.0,
        base_gate_init_bias: float = 4.0,
        residual_gate_init_bias: float = -4.0,
        zero_init_gates: bool = True,
        disable_diffusion_residual: bool = False,
        direct_residual_nodiff: bool = False,
        spatial_artifact_dir: str | None = None,
        spatial_model_config: dict[str, Any] | None = None,
        single_artifact_dir: str | None = None,
        single_model_config: dict[str, Any] | None = None,
        frozen_spatial: FrozenSpatialAdapter | None = None,
        frozen_single: FrozenSingleAdapter | None = None,
    ):
        super().__init__()
        if variant not in VALID_VARIANTS:
            raise ValueError(f"Unsupported ST-DiffTEM variant: {variant}")
        if prediction_type not in VALID_PREDICTION_TYPES:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")
        if residual_inference_mode not in VALID_RESIDUAL_INFERENCE_MODES:
            raise ValueError(f"Unsupported residual_inference_mode: {residual_inference_mode}")
        if residual_anchor_mode not in VALID_RESIDUAL_ANCHOR_MODES:
            raise ValueError(f"Unsupported residual_anchor_mode: {residual_anchor_mode}")
        self.variant = str(variant)
        self.seq_len = int(seq_len)
        self.num_receivers = int(num_receivers)
        self.diffusion_steps = int(diffusion_steps)
        self.sample_steps = int(sample_steps)
        self.sampler_seed = int(sampler_seed)
        self.refine_scale = float(refine_scale)
        self.prediction_type = str(prediction_type)
        self.residual_inference_mode = str(residual_inference_mode)
        self.residual_anchor_mode = str(residual_anchor_mode)
        self.residual_clip_value = float(residual_clip_value)
        self.max_residual_gate = float(max_residual_gate)
        self.rho_max = float(rho_max)
        self.residual_scale_by_w_single = bool(residual_scale_by_w_single)
        self.residual_scale_floor = float(residual_scale_floor)
        self.use_temporal_gate = bool(use_temporal_gate)
        self.temporal_gate_init_bias = float(temporal_gate_init_bias)
        self.base_gate_init_bias = float(base_gate_init_bias)
        self.residual_gate_init_bias = float(residual_gate_init_bias)
        self.zero_init_gates = bool(zero_init_gates)
        self.disable_diffusion_residual = bool(disable_diffusion_residual)
        self.direct_residual_nodiff = bool(direct_residual_nodiff)

        self.spatial_artifact_dir = str(spatial_artifact_dir) if spatial_artifact_dir is not None else None
        self.single_artifact_dir = str(single_artifact_dir) if single_artifact_dir is not None else None
        self.spatial_model_config = dict(spatial_model_config or {})
        self.single_model_config = dict(single_model_config or {})

        self.frozen_spatial = frozen_spatial or self._build_spatial_adapter(
            seq_len=self.seq_len,
            num_receivers=self.num_receivers,
            num_heads=num_heads,
            dropout=dropout,
            spatial_artifact_dir=spatial_artifact_dir,
            spatial_model_config=self.spatial_model_config,
        )

        self.frozen_single: FrozenSingleAdapter | None = None
        if self.variant == "hybrid_base":
            self.frozen_single = frozen_single or self._build_single_adapter(
                seq_len=self.seq_len,
                num_receivers=self.num_receivers,
                num_heads=num_heads,
                single_artifact_dir=single_artifact_dir,
                single_model_config=self.single_model_config,
            )

        if self.variant == "spatial_base":
            self.base_gate_encoder = None
            self.base_gate_net = None
            residual_condition_channels = 4
        else:
            self.base_gate_encoder = ConditionEncoder(
                in_channels=6,
                num_receivers=self.num_receivers,
                hidden_channels=condition_hidden_channels,
                feature_dim=condition_feature_dim,
                spatial_layers=condition_spatial_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.base_gate_net = ReceiverGateNet(
                feature_dim=condition_feature_dim,
                num_receivers=self.num_receivers,
                hidden_dim=gate_hidden_dim,
                dropout=dropout,
                init_bias=self.base_gate_init_bias,
                zero_init_output=self.zero_init_gates,
            )
            residual_condition_channels = 7

        self.residual_encoder = ConditionEncoder(
            in_channels=residual_condition_channels,
            num_receivers=self.num_receivers,
            hidden_channels=condition_hidden_channels,
            feature_dim=condition_feature_dim,
            spatial_layers=condition_spatial_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.residual_gate_net = ReceiverGateNet(
            feature_dim=condition_feature_dim,
            num_receivers=self.num_receivers,
            hidden_dim=gate_hidden_dim,
            dropout=dropout,
            init_bias=self.residual_gate_init_bias,
            zero_init_output=self.zero_init_gates,
        )
        self.temporal_gate_net = (
            TemporalResidualGateNet(
                feature_dim=condition_feature_dim,
                hidden_channels=gate_hidden_dim,
                dropout=dropout,
                init_bias=self.temporal_gate_init_bias,
                zero_init_output=self.zero_init_gates,
            )
            if self.use_temporal_gate
            else None
        )
        self.diffusion_head = ResidualDiffusionHead(
            feature_dim=condition_feature_dim,
            hidden_channels=diffusion_hidden_channels,
            step_dim=step_dim,
            dropout=dropout,
        )

        betas = torch.linspace(beta_start, beta_end, self.diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_bars", alpha_bars, persistent=False)

    def _build_spatial_adapter(
        self,
        seq_len: int,
        num_receivers: int,
        num_heads: int,
        dropout: float,
        spatial_artifact_dir: str | None,
        spatial_model_config: dict[str, Any],
    ) -> FrozenSpatialAdapter:
        artifact_path = Path(spatial_artifact_dir) if spatial_artifact_dir is not None else None
        if artifact_path is not None and artifact_path.exists():
            adapter, resolved_config = FrozenSpatialAdapter.from_artifact_dir(artifact_path)
            if not self.spatial_model_config:
                self.spatial_model_config = dict(resolved_config.get("model", {}))
            return adapter

        cfg = dict(spatial_model_config or {})
        spatial_model = SpatialReceiverPCBiLSTM(
            seq_len=int(cfg.get("seq_len", seq_len)),
            num_receivers=int(cfg.get("num_receivers", num_receivers)),
            filters=int(cfg.get("filters", 16)),
            lstm_hidden=int(cfg.get("lstm_hidden", 32)),
            lstm_layers=int(cfg.get("lstm_layers", 2)),
            num_heads=int(cfg.get("num_heads", num_heads)),
            spatial_layers=int(cfg.get("spatial_layers", 2)),
            dropout=float(cfg.get("dropout", dropout)),
            use_graph_bias=bool(cfg.get("use_graph_bias", True)),
            residual=bool(cfg.get("residual", True)),
        )
        return FrozenSpatialAdapter(spatial_model)

    def _build_single_adapter(
        self,
        seq_len: int,
        num_receivers: int,
        num_heads: int,
        single_artifact_dir: str | None,
        single_model_config: dict[str, Any],
    ) -> FrozenSingleAdapter:
        artifact_path = Path(single_artifact_dir) if single_artifact_dir is not None else None
        if artifact_path is not None and artifact_path.exists():
            adapter, resolved_config = FrozenSingleAdapter.from_artifact_dir(artifact_path)
            if not self.single_model_config:
                self.single_model_config = dict(resolved_config.get("model", {}))
            return adapter

        cfg = dict(single_model_config or {})
        single_model = IndependentReceiverPCBiLSTM(
            seq_len=int(cfg.get("seq_len", seq_len)),
            num_receivers=int(cfg.get("num_receivers", num_receivers)),
            filters=int(cfg.get("filters", 16)),
            lstm_hidden=int(cfg.get("lstm_hidden", 32)),
            lstm_layers=int(cfg.get("lstm_layers", 2)),
            num_heads=int(cfg.get("num_heads", num_heads)),
        )
        return FrozenSingleAdapter(single_model)

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_spatial.eval()
        if self.frozen_single is not None:
            self.frozen_single.eval()
        return self

    def get_config(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "seq_len": self.seq_len,
            "num_receivers": self.num_receivers,
            "condition_feature_dim": self.residual_encoder.feature_dim,
            "condition_hidden_channels": self.residual_encoder.hidden_channels,
            "condition_spatial_layers": len(self.residual_encoder.spatial_blocks),
            "gate_hidden_dim": self.residual_gate_net.hidden_dim,
            "diffusion_hidden_channels": self.diffusion_head.hidden_channels,
            "diffusion_steps": self.diffusion_steps,
            "sample_steps": self.sample_steps,
            "beta_start": float(self.betas[0].item()),
            "beta_end": float(self.betas[-1].item()),
            "step_dim": self.diffusion_head.step_dim,
            "num_heads": self.residual_encoder.num_heads,
            "dropout": self.residual_encoder.dropout,
            "refine_scale": self.refine_scale,
            "sampler_seed": self.sampler_seed,
            "prediction_type": self.prediction_type,
            "residual_inference_mode": self.residual_inference_mode,
            "residual_anchor_mode": self.residual_anchor_mode,
            "residual_clip_value": self.residual_clip_value,
            "max_residual_gate": self.max_residual_gate,
            "rho_max": self.rho_max,
            "residual_scale_by_w_single": self.residual_scale_by_w_single,
            "residual_scale_floor": self.residual_scale_floor,
            "use_temporal_gate": self.use_temporal_gate,
            "temporal_gate_init_bias": self.temporal_gate_init_bias,
            "base_gate_init_bias": self.base_gate_init_bias,
            "residual_gate_init_bias": self.residual_gate_init_bias,
            "zero_init_gates": self.zero_init_gates,
            "disable_diffusion_residual": self.disable_diffusion_residual,
            "direct_residual_nodiff": self.direct_residual_nodiff,
            "spatial_artifact_dir": self.spatial_artifact_dir,
            "single_artifact_dir": self.single_artifact_dir,
            "spatial_model_config": self.spatial_model_config,
            "single_model_config": self.single_model_config,
        }

    def _ddim_schedule(self, device: torch.device) -> list[int]:
        count = max(min(self.sample_steps, self.diffusion_steps), 1)
        if count == 1:
            return [self.diffusion_steps - 1]
        steps = torch.linspace(self.diffusion_steps - 1, 0, count, device=device)
        return [int(round(step.item())) for step in steps]

    def q_sample(self, residual_target: torch.Tensor, step_idx: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = residual_target.shape[0]
        if step_idx is None:
            step_idx = torch.randint(0, self.diffusion_steps, (batch_size,), device=residual_target.device)
        alpha_bar = self.alpha_bars.to(residual_target.device)[step_idx].view(batch_size, 1, 1)
        noise = torch.randn_like(residual_target)
        residual_state = torch.sqrt(alpha_bar) * residual_target + torch.sqrt(1.0 - alpha_bar) * noise
        return step_idx, residual_state, noise

    def predict_x0(self, residual_state: torch.Tensor, step_idx: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars.to(residual_state.device)[step_idx].view(step_idx.shape[0], 1, 1)
        return (residual_state - torch.sqrt(1.0 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar.clamp_min(1e-8))

    def _build_base_gate_stack(
        self,
        noisy_log: torch.Tensor,
        single_base_log: torch.Tensor,
        spatial_base_log: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                noisy_log,
                single_base_log,
                spatial_base_log,
                spatial_base_log - single_base_log,
                (noisy_log - single_base_log).abs(),
                (noisy_log - spatial_base_log).abs(),
            ],
            dim=2,
        )

    def _build_residual_stack(
        self,
        noisy_log: torch.Tensor,
        base_log: torch.Tensor,
        single_base_log: torch.Tensor | None,
        spatial_base_log: torch.Tensor,
    ) -> torch.Tensor:
        if self.variant == "spatial_base":
            observed_residual = noisy_log - base_log
            return torch.stack(
                [noisy_log, base_log, observed_residual, observed_residual.abs()],
                dim=2,
            )

        assert single_base_log is not None
        return torch.stack(
            [
                noisy_log,
                single_base_log,
                spatial_base_log,
                base_log,
                (noisy_log - single_base_log).abs(),
                (noisy_log - spatial_base_log).abs(),
                spatial_base_log - single_base_log,
            ],
            dim=2,
        )

    def _resolve_bases(
        self,
        noisy_log: torch.Tensor,
        spatial_base_log: torch.Tensor | None,
        single_base_log: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if spatial_base_log is None:
            spatial_base_log = self.frozen_spatial(noisy_log)
        else:
            spatial_base_log = spatial_base_log.to(dtype=noisy_log.dtype, device=noisy_log.device)

        if self.variant == "spatial_base":
            return spatial_base_log, None

        if single_base_log is None:
            if self.frozen_single is None:
                raise RuntimeError("Hybrid-base ST-DiffTEM requires a frozen single baseline.")
            single_base_log = self.frozen_single(noisy_log)
        else:
            single_base_log = single_base_log.to(dtype=noisy_log.dtype, device=noisy_log.device)
        return spatial_base_log, single_base_log

    def _compute_base(
        self,
        noisy_log: torch.Tensor,
        spatial_base_log: torch.Tensor,
        single_base_log: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.variant == "spatial_base":
            base_gate = torch.ones(noisy_log.shape[0], self.num_receivers, 1, dtype=noisy_log.dtype, device=noisy_log.device)
            w_single = torch.zeros_like(base_gate)
            return spatial_base_log, base_gate, w_single

        assert single_base_log is not None
        assert self.base_gate_encoder is not None
        assert self.base_gate_net is not None
        gate_stack = self._build_base_gate_stack(noisy_log, single_base_log, spatial_base_log)
        gate_local, gate_spatial = self.base_gate_encoder(gate_stack)
        base_gate = self.base_gate_net(gate_local, gate_spatial)
        w_single = self.rho_max * (1.0 - base_gate)
        base_log = spatial_base_log + w_single * (single_base_log - spatial_base_log)
        return base_log, base_gate, w_single

    def _clip_residual(self, residual: torch.Tensor) -> torch.Tensor:
        if self.residual_clip_value <= 0:
            return residual
        clip = torch.as_tensor(self.residual_clip_value, dtype=residual.dtype, device=residual.device)
        return clip * torch.tanh(residual / clip.clamp_min(1e-8))

    def predict_noise_from_x0(
        self,
        residual_state: torch.Tensor,
        step_idx: torch.Tensor,
        x0_pred: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bars.to(residual_state.device)[step_idx].view(step_idx.shape[0], 1, 1)
        return (residual_state - torch.sqrt(alpha_bar) * x0_pred) / torch.sqrt((1.0 - alpha_bar).clamp_min(1e-8))

    def _build_residual_anchor(self, noisy_log: torch.Tensor, base_log: torch.Tensor) -> torch.Tensor:
        if self.residual_anchor_mode == "zero":
            return torch.zeros_like(base_log)
        if self.residual_anchor_mode == "observed":
            return self._clip_residual(noisy_log - base_log)
        raise RuntimeError(f"Unsupported residual_anchor_mode: {self.residual_anchor_mode}")

    def _predict_residual_from_state(
        self,
        local_feat: torch.Tensor,
        spatial_feat: torch.Tensor,
        noisy_log: torch.Tensor,
        base_log: torch.Tensor,
        residual_state: torch.Tensor,
        step_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_out = self.diffusion_head(
            local_feat,
            spatial_feat,
            noisy_log,
            base_log,
            residual_state,
            step_idx,
        )
        if self.prediction_type == "noise":
            noise_pred = head_out
            residual_pred = self.predict_x0(residual_state, step_idx, noise_pred)
        elif self.prediction_type == "x0":
            residual_pred = head_out
            noise_pred = self.predict_noise_from_x0(residual_state, step_idx, residual_pred)
        else:
            raise RuntimeError(f"Unsupported prediction_type: {self.prediction_type}")
        return self._clip_residual(residual_pred), noise_pred

    def predict_anchor_residual(
        self,
        local_feat: torch.Tensor,
        spatial_feat: torch.Tensor,
        noisy_log: torch.Tensor,
        base_log: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_state = self._build_residual_anchor(noisy_log, base_log)
        step_tensor = torch.zeros((noisy_log.shape[0],), device=noisy_log.device, dtype=torch.long)
        return self._predict_residual_from_state(
            local_feat,
            spatial_feat,
            noisy_log,
            base_log,
            anchor_state,
            step_tensor,
        )

    def sample_residual(
        self,
        local_feat: torch.Tensor,
        spatial_feat: torch.Tensor,
        noisy_log: torch.Tensor,
        base_log: torch.Tensor,
    ) -> torch.Tensor:
        if self.residual_inference_mode == "anchor":
            residual_pred, _ = self.predict_anchor_residual(local_feat, spatial_feat, noisy_log, base_log)
            return residual_pred

        schedule = self._ddim_schedule(noisy_log.device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.sampler_seed)
        residual_state = torch.randn(base_log.shape, generator=generator, dtype=base_log.dtype, device="cpu").to(
            noisy_log.device
        )
        for idx, step in enumerate(schedule):
            step_tensor = torch.full((noisy_log.shape[0],), step, device=noisy_log.device, dtype=torch.long)
            x0_pred, noise_pred = self._predict_residual_from_state(
                local_feat,
                spatial_feat,
                noisy_log,
                base_log,
                residual_state,
                step_tensor,
            )
            if idx == len(schedule) - 1:
                residual_state = x0_pred
                break
            prev_step = schedule[idx + 1]
            alpha_bar_prev = self.alpha_bars.to(noisy_log.device)[prev_step]
            residual_state = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1.0 - alpha_bar_prev) * noise_pred
        return residual_state

    def forward(
        self,
        noisy_log: torch.Tensor,
        clean_log: torch.Tensor | None = None,
        spatial_base_log: torch.Tensor | None = None,
        single_base_log: torch.Tensor | None = None,
        base_log: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        if noisy_log.ndim != 3:
            raise ValueError(f"Expected noisy_log shape [B, R, T], got {tuple(noisy_log.shape)}")
        if noisy_log.shape[1] != self.num_receivers or noisy_log.shape[2] != self.seq_len:
            raise ValueError(
                f"Expected noisy_log shape (*, {self.num_receivers}, {self.seq_len}), got {tuple(noisy_log.shape)}"
            )

        spatial_base_log, single_base_log = self._resolve_bases(noisy_log, spatial_base_log, single_base_log)
        if base_log is None:
            base_log, base_gate, w_single = self._compute_base(noisy_log, spatial_base_log, single_base_log)
            anchor_log = base_log
        else:
            base_log = base_log.to(dtype=noisy_log.dtype, device=noisy_log.device)
            anchor_log = base_log
            if self.variant == "spatial_base":
                base_gate = torch.ones(noisy_log.shape[0], self.num_receivers, 1, dtype=noisy_log.dtype, device=noisy_log.device)
                w_single = torch.zeros_like(base_gate)
            else:
                assert single_base_log is not None
                _, base_gate, w_single = self._compute_base(noisy_log, spatial_base_log, single_base_log)

        residual_stack = self._build_residual_stack(noisy_log, base_log, single_base_log, spatial_base_log)
        local_feat, spatial_feat = self.residual_encoder(residual_stack)
        receiver_gate = self.residual_gate_net(local_feat, spatial_feat)
        if self.max_residual_gate >= 0:
            receiver_gate = receiver_gate * self.max_residual_gate
        temporal_gate = None
        if self.temporal_gate_net is not None:
            temporal_gate = self.temporal_gate_net(local_feat, spatial_feat, noisy_log, base_log)
            residual_gate = receiver_gate * temporal_gate
        else:
            residual_gate = receiver_gate

        if self.variant == "hybrid_base":
            rho_denom = max(self.rho_max, 1e-8)
            w_single_norm = torch.clamp(w_single / rho_denom, 0.0, 1.0)
        else:
            w_single_norm = None

        residual_target = clean_log - base_log if clean_log is not None else None
        if self.disable_diffusion_residual:
            diffusion_residual_pred = None
            anchor_residual_pred = None
            anchor_noise_pred = None
            noise = None
            noise_pred = None
            residual_pred = torch.zeros_like(base_log)
            effective_residual_gate = torch.zeros_like(residual_gate)
            residual_scale_factor = torch.zeros_like(residual_gate)
            if self.variant == "hybrid_base" and w_single_norm is None:
                w_single_norm = torch.zeros_like(w_single)
        elif self.direct_residual_nodiff and clean_log is not None and self.training:
            assert residual_target is not None
            step_idx = torch.zeros((noisy_log.shape[0],), device=noisy_log.device, dtype=torch.long)
            residual_state = self._build_residual_anchor(noisy_log, base_log)
            diffusion_residual_pred, noise_pred = self._predict_residual_from_state(
                local_feat,
                spatial_feat,
                noisy_log,
                base_log,
                residual_state,
                step_idx,
            )
            anchor_residual_pred = None
            anchor_noise_pred = None
            noise = None
            residual_pred = diffusion_residual_pred
        elif clean_log is not None and self.training:
            assert residual_target is not None
            step_idx, residual_state, noise = self.q_sample(residual_target)
            diffusion_residual_pred, noise_pred = self._predict_residual_from_state(
                local_feat,
                spatial_feat,
                noisy_log,
                base_log,
                residual_state,
                step_idx,
            )
            anchor_residual_pred = None
            anchor_noise_pred = None
            if self.prediction_type == "x0" and self.residual_inference_mode == "anchor":
                anchor_residual_pred, anchor_noise_pred = self.predict_anchor_residual(
                    local_feat,
                    spatial_feat,
                    noisy_log,
                    base_log,
                )
                residual_pred = anchor_residual_pred
            else:
                residual_pred = diffusion_residual_pred
        else:
            diffusion_residual_pred = None
            anchor_residual_pred = None
            anchor_noise_pred = None
            noise = None
            noise_pred = None
            residual_pred = self.sample_residual(local_feat, spatial_feat, noisy_log, base_log)

        if self.disable_diffusion_residual:
            pred_log = base_log
        else:
            if self.variant == "hybrid_base" and self.residual_scale_by_w_single:
                assert w_single_norm is not None
                residual_scale_factor = self.residual_scale_floor + (1.0 - self.residual_scale_floor) * w_single_norm
                effective_residual_gate = residual_gate * residual_scale_factor
            else:
                residual_scale_factor = torch.ones_like(residual_gate)
                effective_residual_gate = residual_gate
            pred_log = base_log + self.refine_scale * effective_residual_gate * residual_pred
        if not return_aux:
            return pred_log
        return pred_log, {
            "variant": self.variant,
            "base_log": base_log,
            "anchor_log": anchor_log,
            "spatial_base_log": spatial_base_log,
            "single_base_log": single_base_log,
            "base_gate": base_gate,
            "w_single": w_single,
            "rho_max": base_log.new_tensor(float(self.rho_max)),
            "gate": residual_gate,
            "effective_residual_gate": effective_residual_gate,
            "residual_scale_factor": residual_scale_factor,
            "w_single_norm": w_single_norm,
            "receiver_gate": receiver_gate,
            "temporal_gate": temporal_gate,
            "residual_pred": residual_pred,
            "residual_target": residual_target,
            "diffusion_residual_pred": diffusion_residual_pred,
            "anchor_residual_pred": anchor_residual_pred,
            "anchor_noise_pred": anchor_noise_pred,
            "noise_target": noise,
            "noise_pred": noise_pred,
            "prediction_type": self.prediction_type,
            "direct_residual_nodiff": self.direct_residual_nodiff,
        }

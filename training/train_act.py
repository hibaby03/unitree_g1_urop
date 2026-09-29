#!/usr/bin/env python3
"""Train an ACT (Action Chunking with Transformers) policy on active-stereo episodes.

Input is the episode directory written by ``host/teleop_record_dex3_tactile.py``
(xr_teleoperate ``EpisodeWriter`` layout)::

    <data-dir>/episode_0000/data.json
    <data-dir>/episode_0000/colors/000000_color_0.jpg   # ZED left
    <data-dir>/episode_0000/colors/000000_color_1.jpg   # ZED right

Observation: ZED left/right images + joint state
(left_arm 7, right_arm 7, left_ee 7, right_ee 7, neck 2 = 30).
Action: the commanded targets in the same order, so the policy also moves
the active camera (neck yaw/pitch), as in the active-camera-only setup.

Model follows Zhao et al. 2023 (ACT): ResNet-18 backbone per camera,
CVAE encoder over the action chunk, transformer encoder/decoder that
predicts ``chunk_size`` future actions. Loss = masked L1 + kl_weight * KL.

Example::

    python training/train_act.py \
        --data-dir ~/xr_teleoperate/teleop/utils/data/active_stereo_task \
        --out-dir runs/act_active_stereo
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("train_act")

CAMERA_KEYS = ("color_0", "color_1")  # ZED left, right
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# (group key, dof). Order defines the state/action vector layout.
JOINT_GROUPS = (("left_arm", 7), ("right_arm", 7), ("left_ee", 7), ("right_ee", 7))
NECK_GROUP = ("neck", 2)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _forward_back_fill(rows: list[Optional[list[float]]], dof: int, what: str) -> np.ndarray:
    """Fill missing (empty) rows with the previous valid row, then the next one."""

    valid = [i for i, row in enumerate(rows) if row is not None and len(row) == dof]
    if not valid:
        raise ValueError(f"{what}: no valid values in episode")
    out = np.zeros((len(rows), dof), dtype=np.float32)
    last = rows[valid[0]]
    for i, row in enumerate(rows):
        if row is not None and len(row) == dof:
            last = row
        out[i] = last
    return out


@dataclass
class Episode:
    root: Path
    image_paths: list[dict[str, str]]
    qpos: np.ndarray    # (T, state_dim)
    action: np.ndarray  # (T, action_dim)


def load_episode(episode_dir: Path, groups: Sequence[tuple[str, int]],
                 camera_keys: Sequence[str]) -> Episode:
    with open(episode_dir / "data.json", "r", encoding="utf-8") as f:
        items = json.load(f)["data"]
    if not items:
        raise ValueError(f"{episode_dir}: empty episode")

    state_cols, action_cols = [], []
    for key, dof in groups:
        states = [item["states"].get(key, {}).get("qpos") or None for item in items]
        actions = [item["actions"].get(key, {}).get("qpos") or None for item in items]
        state_cols.append(_forward_back_fill(states, dof, f"{episode_dir.name} states.{key}"))
        action_cols.append(_forward_back_fill(actions, dof, f"{episode_dir.name} actions.{key}"))

    # The recorder writes colors={} when the head frame was None; reuse the
    # nearest earlier (or first later) frame for those items.
    frames = [(item.get("colors") or {}) for item in items]
    frames = [{k: c[k] for k in camera_keys} if all(k in c for k in camera_keys) else None
              for c in frames]
    num_missing = sum(f is None for f in frames)
    if num_missing == len(frames):
        raise ValueError(f"{episode_dir.name}: no frames with all of {list(camera_keys)}")
    if num_missing:
        logger.warning("%s: %d/%d frames without images, filled from neighbours",
                       episode_dir.name, num_missing, len(frames))
    last = next(f for f in frames if f is not None)
    image_paths = []
    for frame in frames:
        last = frame or last
        image_paths.append(last)

    return Episode(episode_dir, image_paths,
                   np.concatenate(state_cols, axis=1), np.concatenate(action_cols, axis=1))


def find_episodes(data_dir: Path) -> list[Path]:
    episodes = sorted(p.parent for p in data_dir.glob("*/data.json"))
    if not episodes:
        raise FileNotFoundError(f"no */data.json under {data_dir}")
    return episodes


@dataclass
class NormStats:
    qpos_mean: np.ndarray
    qpos_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @classmethod
    def from_episodes(cls, episodes: Sequence[Episode]) -> "NormStats":
        qpos = np.concatenate([e.qpos for e in episodes])
        action = np.concatenate([e.action for e in episodes])
        return cls(qpos.mean(0), np.clip(qpos.std(0), 1e-2, None),
                   action.mean(0), np.clip(action.std(0), 1e-2, None))

    def to_dict(self) -> dict[str, list[float]]:
        return {k: v.tolist() for k, v in asdict(self).items()}


class ACTDataset(Dataset):
    """One sample = (images at t, qpos at t, actions[t:t+chunk], pad mask)."""

    def __init__(self, episodes: Sequence[Episode], stats: NormStats, chunk_size: int,
                 image_size: tuple[int, int], camera_keys: Sequence[str],
                 augment: bool = False) -> None:
        self.episodes = list(episodes)
        self.stats = stats
        self.chunk_size = chunk_size
        self.image_size = image_size  # (H, W)
        self.camera_keys = tuple(camera_keys)
        self.augment = augment
        self.index = [(e, t) for e, ep in enumerate(self.episodes) for t in range(len(ep.qpos))]

    def __len__(self) -> int:
        return len(self.index)

    def _load_image(self, path: Path) -> torch.Tensor:
        # EpisodeWriter saves with cv2.imwrite(BGR), so the JPEG decodes as RGB.
        h, w = self.image_size
        img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
        tensor = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        if self.augment:
            tensor = tensor * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1)
            tensor = tensor.clamp(0.0, 1.0)
        return tensor

    def __getitem__(self, i: int):
        e, t = self.index[i]
        ep = self.episodes[e]
        images = torch.stack([self._load_image(ep.root / ep.image_paths[t][k])
                              for k in self.camera_keys])

        chunk = ep.action[t:t + self.chunk_size]
        pad = self.chunk_size - len(chunk)
        is_pad = np.zeros(self.chunk_size, dtype=bool)
        if pad:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, axis=0)])
            is_pad[-pad:] = True

        qpos = (ep.qpos[t] - self.stats.qpos_mean) / self.stats.qpos_std
        action = (chunk - self.stats.action_mean) / self.stats.action_std
        return (images, torch.from_numpy(qpos.astype(np.float32)),
                torch.from_numpy(action.astype(np.float32)), torch.from_numpy(is_pad))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def sine_position_embedding_2d(h: int, w: int, dim: int, device) -> torch.Tensor:
    """DETR-style 2D sine embedding, shape (h*w, dim)."""

    half = dim // 2
    y = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1).expand(h, w)
    x = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(0).expand(h, w)
    y = y / max(h - 1, 1) * 2 * math.pi
    x = x / max(w - 1, 1) * 2 * math.pi
    dim_t = 10000 ** (2 * (torch.arange(half, device=device) // 2) / half)
    pos_x, pos_y = x[..., None] / dim_t, y[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(2)
    return torch.cat((pos_y, pos_x), dim=-1).reshape(h * w, dim)


def sine_position_embedding_1d(n: int, dim: int) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    table = torch.zeros(n, dim)
    table[:, 0::2] = torch.sin(pos * div)
    table[:, 1::2] = torch.cos(pos * div)
    return table


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        import torchvision

        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = torchvision.models.resnet18(weights=weights,
                                             norm_layer=torchvision.ops.FrozenBatchNorm2d)
        self.body = nn.Sequential(*list(resnet.children())[:-2])
        self.num_channels = 512
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body((x - self.mean) / self.std)


@dataclass
class ACTConfig:
    state_dim: int
    action_dim: int
    num_cameras: int
    chunk_size: int = 100
    hidden_dim: int = 512
    dim_feedforward: int = 3200
    nheads: int = 8
    enc_layers: int = 4
    dec_layers: int = 7
    latent_dim: int = 32
    dropout: float = 0.1
    pretrained_backbone: bool = True


class ACT(nn.Module):
    def __init__(self, cfg: ACTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        def encoder(layers: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(d, cfg.nheads, cfg.dim_feedforward,
                                               cfg.dropout, batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

        # CVAE encoder: [CLS, qpos, a_1..a_k] -> (mu, logvar)
        self.cvae_encoder = encoder(cfg.enc_layers)
        self.cls_embed = nn.Parameter(torch.zeros(1, 1, d))
        self.cvae_qpos_proj = nn.Linear(cfg.state_dim, d)
        self.cvae_action_proj = nn.Linear(cfg.action_dim, d)
        self.latent_proj = nn.Linear(d, 2 * cfg.latent_dim)
        self.register_buffer("cvae_pos", sine_position_embedding_1d(cfg.chunk_size + 2, d))

        # Observation encoder / action decoder
        self.backbone = ResNetBackbone(cfg.pretrained_backbone)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, d, kernel_size=1)
        self.camera_embed = nn.Embedding(cfg.num_cameras, d)
        self.latent_out_proj = nn.Linear(cfg.latent_dim, d)
        self.qpos_proj = nn.Linear(cfg.state_dim, d)
        self.extra_pos = nn.Embedding(2, d)  # latent, qpos tokens
        self.encoder = encoder(cfg.enc_layers)
        dec_layer = nn.TransformerDecoderLayer(d, cfg.nheads, cfg.dim_feedforward,
                                               cfg.dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.dec_layers, norm=nn.LayerNorm(d))
        self.query_embed = nn.Embedding(cfg.chunk_size, d)
        self.action_head = nn.Linear(d, cfg.action_dim)

    def encode_latent(self, qpos, actions, is_pad):
        b = qpos.shape[0]
        tokens = torch.cat([self.cls_embed.expand(b, -1, -1),
                            self.cvae_qpos_proj(qpos).unsqueeze(1),
                            self.cvae_action_proj(actions)], dim=1)
        tokens = tokens + self.cvae_pos.unsqueeze(0)
        pad = torch.cat([torch.zeros(b, 2, dtype=torch.bool, device=qpos.device), is_pad], dim=1)
        out = self.cvae_encoder(tokens, src_key_padding_mask=pad)[:, 0]
        mu, logvar = self.latent_proj(out).chunk(2, dim=-1)
        return mu, logvar

    def forward(self, images, qpos, actions=None, is_pad=None):
        """images (B, N, 3, H, W) in [0, 1]; qpos (B, S) normalized.

        Training (actions given): returns (pred, mu, logvar).
        Inference: returns (pred, None, None) with z = 0.
        """

        b, n = images.shape[:2]
        if actions is not None:
            mu, logvar = self.encode_latent(qpos, actions, is_pad)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            mu = logvar = None
            z = torch.zeros(b, self.cfg.latent_dim, device=qpos.device)

        feats = self.input_proj(self.backbone(images.flatten(0, 1)))  # (B*N, d, h, w)
        _, d, h, w = feats.shape
        feats = feats.view(b, n, d, h * w).permute(0, 1, 3, 2)          # (B, N, hw, d)
        pos = sine_position_embedding_2d(h, w, d, images.device)
        feats = feats + pos + self.camera_embed.weight[:n, None, :]
        feats = feats.reshape(b, n * h * w, d)

        extra = torch.stack([self.latent_out_proj(z), self.qpos_proj(qpos)], dim=1)
        extra = extra + self.extra_pos.weight.unsqueeze(0)
        memory = self.encoder(torch.cat([extra, feats], dim=1))

        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        out = self.decoder(queries, memory)
        return self.action_head(out), mu, logvar


def act_loss(pred, target, is_pad, mu, logvar, kl_weight: float):
    l1 = (F.l1_loss(pred, target, reduction="none") * (~is_pad).unsqueeze(-1)).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
    return l1 + kl_weight * kl, {"l1": l1.item(), "kl": kl.item()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ACT on active-stereo G1 episodes.")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="task directory containing episode_*/data.json")
    p.add_argument("--out-dir", type=Path, default=Path("runs/act"))
    p.add_argument("--no-neck", action="store_true",
                   help="exclude neck yaw/pitch from state/action (static camera baseline)")
    p.add_argument("--cameras", nargs="+", default=list(CAMERA_KEYS),
                   help="color keys to use (default: ZED left and right)")
    p.add_argument("--image-size", type=int, nargs=2, default=(240, 424), metavar=("H", "W"))
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dim-feedforward", type=int, default=3200)
    p.add_argument("--enc-layers", type=int, default=4)
    p.add_argument("--dec-layers", type=int, default=7)
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-pretrained", action="store_true",
                   help="do not download ImageNet ResNet-18 weights")
    p.add_argument("--kl-weight", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=100, help="epochs between checkpoints")
    p.add_argument("--augment", action="store_true", help="brightness/contrast jitter")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=Path, help="checkpoint to resume from")
    return p.parse_args(argv)


def save_checkpoint(path: Path, model: ACT, optimizer, epoch: int, best_val: float,
                    stats: NormStats, groups, camera_keys, image_size) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "config": asdict(model.cfg),
        "norm_stats": stats.to_dict(),
        "joint_groups": [list(g) for g in groups],
        "camera_keys": list(camera_keys),
        "image_size": list(image_size),
    }, path)


def run_epoch(model, loader, device, kl_weight, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "l1": 0.0, "kl": 0.0}
    count = 0
    with torch.set_grad_enabled(training):
        for images, qpos, actions, is_pad in loader:
            images, qpos = images.to(device), qpos.to(device)
            actions, is_pad = actions.to(device), is_pad.to(device)
            pred, mu, logvar = model(images, qpos, actions, is_pad)
            loss, parts = act_loss(pred, actions, is_pad, mu, logvar, kl_weight)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            n = qpos.shape[0]
            totals["loss"] += loss.item() * n
            totals["l1"] += parts["l1"] * n
            totals["kl"] += parts["kl"] * n
            count += n
    return {k: v / max(count, 1) for k, v in totals.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    groups = list(JOINT_GROUPS) + ([] if args.no_neck else [NECK_GROUP])
    episode_dirs = find_episodes(args.data_dir.expanduser())
    episodes = []
    for path in episode_dirs:
        try:
            episodes.append(load_episode(path, groups, args.cameras))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.warning("skipping %s: %s", path.name, exc)
    if not episodes:
        raise SystemExit("no usable episodes")
    random.Random(args.seed).shuffle(episodes)
    num_val = int(round(len(episodes) * args.val_ratio)) if len(episodes) > 1 else 0
    val_eps, train_eps = episodes[:num_val], episodes[num_val:]
    stats = NormStats.from_episodes(train_eps)
    logger.info("episodes: %d train / %d val, frames: %d, state/action dim: %d, device: %s",
                len(train_eps), len(val_eps), sum(len(e.qpos) for e in episodes),
                train_eps[0].qpos.shape[1], device)

    image_size = tuple(args.image_size)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(ACTDataset(train_eps, stats, args.chunk_size, image_size,
                                         args.cameras, augment=args.augment),
                              shuffle=True, drop_last=False, **loader_kw)
    val_loader = (DataLoader(ACTDataset(val_eps, stats, args.chunk_size, image_size,
                                        args.cameras), shuffle=False, **loader_kw)
                  if val_eps else None)

    cfg = ACTConfig(state_dim=train_eps[0].qpos.shape[1], action_dim=train_eps[0].action.shape[1],
                    num_cameras=len(args.cameras), chunk_size=args.chunk_size,
                    hidden_dim=args.hidden_dim, dim_feedforward=args.dim_feedforward,
                    nheads=args.nheads, enc_layers=args.enc_layers, dec_layers=args.dec_layers,
                    latent_dim=args.latent_dim, dropout=args.dropout,
                    pretrained_backbone=not args.no_pretrained)
    model = ACT(cfg).to(device)
    backbone_params = [p for n, p in model.named_parameters()
                       if n.startswith("backbone.") and p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith("backbone.") and p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": other_params, "lr": args.lr},
                                   {"params": backbone_params, "lr": args.lr_backbone}],
                                  weight_decay=args.weight_decay)
    logger.info("parameters: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    start_epoch, best_val = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best_val = ckpt["epoch"] + 1, ckpt["best_val"]
        logger.info("resumed from %s (epoch %d)", args.resume, ckpt["epoch"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  f, indent=2)
    ckpt_kw = dict(stats=stats, groups=groups, camera_keys=args.cameras, image_size=image_size)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train = run_epoch(model, train_loader, device, args.kl_weight, optimizer)
        msg = f"epoch {epoch:4d} train loss {train['loss']:.4f} l1 {train['l1']:.4f} kl {train['kl']:.4f}"
        if val_loader is not None:
            val = run_epoch(model, val_loader, device, args.kl_weight)
            msg += f" | val loss {val['loss']:.4f} l1 {val['l1']:.4f}"
            if val["loss"] < best_val:
                best_val = val["loss"]
                save_checkpoint(args.out_dir / "policy_best.ckpt", model, optimizer, epoch,
                                best_val, **ckpt_kw)
                msg += " *"
        logger.info("%s (%.1fs)", msg, time.time() - t0)
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(args.out_dir / f"policy_epoch_{epoch + 1}.ckpt", model, optimizer,
                            epoch, best_val, **ckpt_kw)
        save_checkpoint(args.out_dir / "policy_last.ckpt", model, optimizer, epoch,
                        best_val, **ckpt_kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

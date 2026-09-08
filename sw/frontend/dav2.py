# ABOUTME: Builds the Depth Anything V2 graph (DINOv2 ViT encoder + DPT depth head) into the toolchain IR.
# ABOUTME: Structure is derived from the checkpoint's own tensor names and shapes, not from a copy of the reference repo.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..ir import Graph, GraphBuilder
from ..kernels_float import interpolate_bicubic

DEFAULT_CHECKPOINT = Path(
    "~/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Small/"
    "snapshots/03876f8651c73a60fe4c2c48294e09fcb6838fcf/depth_anything_v2_vits.pth"
).expanduser()


@dataclass(frozen=True)
class DAV2Config:
    """Encoder geometry. Only ViT-S is wired up; the others differ in these numbers alone."""

    encoder: str = "vits"
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    patch_size: int = 14
    img_size: int = 518
    features: int = 64
    out_channels: tuple[int, ...] = (48, 96, 192, 384)
    intermediate_layers: tuple[int, ...] = (2, 5, 8, 11)
    layernorm_eps: float = 1e-6
    pos_embed_grid: int = 37  # sqrt(1370 - 1)

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def patch_grid(self) -> int:
        if self.img_size % self.patch_size:
            raise ValueError(f"img_size {self.img_size} is not a multiple of patch {self.patch_size}")
        return self.img_size // self.patch_size

    @property
    def num_tokens(self) -> int:
        return self.patch_grid**2 + 1


def load_state_dict(path: Path | str = DEFAULT_CHECKPOINT) -> dict[str, np.ndarray]:
    """Read the .pth into plain numpy so nothing downstream depends on torch."""
    import torch  # local import: only the frontend needs torch, the emulator does not

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError(f"expected a state dict in {path}, got {type(raw)}")
    return {k: v.detach().cpu().float().numpy() for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    b: GraphBuilder
    sd: dict[str, np.ndarray]
    cfg: DAV2Config
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def w(self, key: str) -> np.ndarray:
        if key not in self.sd:
            raise KeyError(f"checkpoint is missing {key!r}; is this a Depth-Anything-V2 file?")
        return self.sd[key]

    def const(self, key: str) -> str:
        return self.b.constant(key.replace(".", "_"), self.w(key))

    def emit(self, op: str, ins: list[str], stem: str, out_shape: tuple[int, ...], **attrs) -> str:
        out = self.b.emit(op, ins, stem, **attrs)
        self.shapes[out] = out_shape
        return out

    # -- shape-aware sugar ------------------------------------------------

    def conv(self, x: str, prefix: str, stem: str, stride: int = 1, pad: int = 0,
             bias: bool = True) -> str:
        w = self.w(f"{prefix}.weight")
        ins = [x, self.const(f"{prefix}.weight")]
        if bias:
            ins.append(self.const(f"{prefix}.bias"))
        n, _, h, ww = self.shapes[x]
        oh = (h + 2 * pad - w.shape[2]) // stride + 1
        ow = (ww + 2 * pad - w.shape[3]) // stride + 1
        return self.emit("conv2d", ins, stem, (n, w.shape[0], oh, ow), stride=stride, pad=pad)

    def conv_t(self, x: str, prefix: str, stem: str, stride: int, pad: int = 0) -> str:
        w = self.w(f"{prefix}.weight")
        ins = [x, self.const(f"{prefix}.weight"), self.const(f"{prefix}.bias")]
        n, _, h, ww = self.shapes[x]
        oh = (h - 1) * stride - 2 * pad + w.shape[2]
        ow = (ww - 1) * stride - 2 * pad + w.shape[3]
        return self.emit("conv_transpose2d", ins, stem, (n, w.shape[1], oh, ow),
                         stride=stride, pad=pad)

    def matmul(self, x: str, prefix: str, stem: str, bias: bool = True) -> str:
        w = self.w(f"{prefix}.weight")
        ins = [x, self.const(f"{prefix}.weight")]
        if bias:
            ins.append(self.const(f"{prefix}.bias"))
        shape = self.shapes[x][:-1] + (w.shape[0],)
        return self.emit("matmul", ins, stem, shape)

    def matmul_arrays(self, x: str, w: np.ndarray, bias: np.ndarray | None, stem: str) -> str:
        """A matmul against weights derived at compile time rather than read verbatim."""
        ins = [x, self.b.constant(f"{stem.replace('/', '_')}_w", w)]
        if bias is not None:
            ins.append(self.b.constant(f"{stem.replace('/', '_')}_b", bias))
        return self.emit("matmul", ins, stem, self.shapes[x][:-1] + (w.shape[0],))

    def layernorm(self, x: str, prefix: str, stem: str) -> str:
        ins = [x, self.const(f"{prefix}.weight"), self.const(f"{prefix}.bias")]
        return self.emit("layernorm", ins, stem, self.shapes[x],
                         axis=-1, eps=self.cfg.layernorm_eps)

    def reshape(self, x: str, shape: tuple[int, ...], stem: str) -> str:
        return self.emit("reshape", [x], stem, shape, shape=list(shape))

    def transpose(self, x: str, perm: tuple[int, ...], stem: str) -> str:
        shape = tuple(self.shapes[x][p] for p in perm)
        return self.emit("transpose", [x], stem, shape, perm=list(perm))

    def add(self, a: str, b: str, stem: str) -> str:
        shape = self.shapes.get(a) or self.shapes[b]
        return self.emit("add", [a, b], stem, shape)

    def interp(self, x: str, size: tuple[int, int], stem: str) -> str:
        n, c, _, _ = self.shapes[x]
        return self.emit("interpolate", [x], stem, (n, c, *size),
                         mode="bilinear", size=list(size), align_corners=True)

    def relu(self, x: str, stem: str) -> str:
        return self.emit("relu", [x], stem, self.shapes[x])


def _resample_pos_embed(pos: np.ndarray, cfg: DAV2Config) -> np.ndarray:
    """DINOv2 resamples the patch grid bicubically when the input is not 518x518."""
    grid = cfg.patch_grid
    if grid == cfg.pos_embed_grid:
        return pos
    cls_pos = pos[:, :1]
    patch_pos = pos[:, 1:]
    m = int(math.sqrt(patch_pos.shape[1]))
    nchw = patch_pos.reshape(1, m, m, -1).transpose(0, 3, 1, 2)
    resized = interpolate_bicubic(nchw, (grid, grid), align_corners=False)
    resized = resized.transpose(0, 2, 3, 1).reshape(1, grid * grid, -1)
    return np.concatenate([cls_pos, resized], axis=1).astype(np.float32)


def _attention(c: _Ctx, x: str, prefix: str, tag: str) -> str:
    cfg = c.cfg
    t = c.shapes[x][1]
    h, d = cfg.num_heads, cfg.head_dim
    e = cfg.embed_dim

    # The checkpoint fuses q, k and v into one [3E, E] projection. Emitting it
    # fused and slicing would force all three onto a single quantization scale,
    # because a slice preserves values and therefore preserves scale -- and v's
    # range is far narrower than q's, so v would lose most of its resolution.
    # Splitting the weight is exact and gives each its own scale.
    w_qkv = c.w(f"{prefix}.attn.qkv.weight")
    b_qkv = c.w(f"{prefix}.attn.qkv.bias")

    heads: list[str] = []
    for i, part in enumerate(("q", "k", "v")):
        w = w_qkv[i * e:(i + 1) * e]
        bias = b_qkv[i * e:(i + 1) * e]
        if part == "q":
            # DINOv2 scales q by head_dim**-0.5 before the qk product. Folding it
            # into the projection is exact, removes an op, and lets q be
            # calibrated at its post-scale range.
            w = w * (d**-0.5)
            bias = bias * (d**-0.5)
        s = c.matmul_arrays(x, w, bias, f"{tag}/{part}")
        s = c.reshape(s, (1, t, h, d), f"{tag}/{part}_split")
        heads.append(c.transpose(s, (0, 2, 1, 3), f"{tag}/{part}_heads"))
    q, k, v = heads

    kt = c.transpose(k, (0, 1, 3, 2), f"{tag}/k_t")
    logits = c.emit("batch_matmul", [q, kt], f"{tag}/logits", (1, h, t, t))
    probs = c.emit("softmax", [logits], f"{tag}/probs", (1, h, t, t), axis=-1)
    ctx = c.emit("batch_matmul", [probs, v], f"{tag}/ctx", (1, h, t, d))
    ctx = c.transpose(ctx, (0, 2, 1, 3), f"{tag}/ctx_t")
    ctx = c.reshape(ctx, (1, t, cfg.embed_dim), f"{tag}/ctx_flat")
    return c.matmul(ctx, f"{prefix}.attn.proj", f"{tag}/attn_out")


def _block(c: _Ctx, x: str, index: int) -> str:
    prefix = f"pretrained.blocks.{index}"
    tag = f"blk{index}"

    h = c.layernorm(x, f"{prefix}.norm1", f"{tag}/norm1")
    h = _attention(c, h, prefix, tag)
    ls1 = c.b.constant(f"{tag}_ls1_gamma", c.w(f"{prefix}.ls1.gamma"))
    h = c.emit("mul", [h, ls1], f"{tag}/ls1", c.shapes[h])
    x = c.add(x, h, f"{tag}/res1")

    h = c.layernorm(x, f"{prefix}.norm2", f"{tag}/norm2")
    h = c.matmul(h, f"{prefix}.mlp.fc1", f"{tag}/fc1")
    h = c.emit("gelu", [h], f"{tag}/act", c.shapes[h])
    h = c.matmul(h, f"{prefix}.mlp.fc2", f"{tag}/fc2")
    ls2 = c.b.constant(f"{tag}_ls2_gamma", c.w(f"{prefix}.ls2.gamma"))
    h = c.emit("mul", [h, ls2], f"{tag}/ls2", c.shapes[h])
    return c.add(x, h, f"{tag}/res2")


def _residual_conv_unit(c: _Ctx, x: str, prefix: str, tag: str) -> str:
    h = c.relu(x, f"{tag}/relu1")
    h = c.conv(h, f"{prefix}.conv1", f"{tag}/conv1", pad=1)
    h = c.relu(h, f"{tag}/relu2")
    h = c.conv(h, f"{prefix}.conv2", f"{tag}/conv2", pad=1)
    return c.add(h, x, f"{tag}/rcu_out")


def _fusion_block(c: _Ctx, deep: str, skip: str | None, prefix: str, tag: str,
                  size: tuple[int, int]) -> str:
    out = deep
    if skip is not None:
        res = _residual_conv_unit(c, skip, f"{prefix}.resConfUnit1", f"{tag}/rcu1")
        out = c.add(out, res, f"{tag}/fuse")
    out = _residual_conv_unit(c, out, f"{prefix}.resConfUnit2", f"{tag}/rcu2")
    out = c.interp(out, size, f"{tag}/up")
    return c.conv(out, f"{prefix}.out_conv", f"{tag}/out_conv")


def build_dav2_graph(state_dict: dict[str, np.ndarray] | None = None,
                     cfg: DAV2Config | None = None) -> Graph:
    """Construct the full DA-V2 inference graph. Returns a validated IR graph."""
    cfg = cfg or DAV2Config()
    sd = state_dict if state_dict is not None else load_state_dict()
    b = GraphBuilder(f"depth_anything_v2_{cfg.encoder}_{cfg.img_size}")
    c = _Ctx(b=b, sd=sd, cfg=cfg)

    grid = cfg.patch_grid
    n_patch = grid * grid
    img = b.input("image")
    c.shapes[img] = (1, 3, cfg.img_size, cfg.img_size)

    # -- patch embedding -------------------------------------------------
    x = c.conv(img, "pretrained.patch_embed.proj", "patch_embed", stride=cfg.patch_size)
    x = c.reshape(x, (1, cfg.embed_dim, n_patch), "patch_flat")
    x = c.transpose(x, (0, 2, 1), "patch_tokens")
    cls = b.constant("cls_token", c.w("pretrained.cls_token"))
    c.shapes[cls] = (1, 1, cfg.embed_dim)
    x = c.emit("concat", [cls, x], "tokens", (1, cfg.num_tokens, cfg.embed_dim), axis=1)
    pos = b.constant("pos_embed", _resample_pos_embed(c.w("pretrained.pos_embed"), cfg))
    x = c.add(x, pos, "tokens_pos")

    # -- transformer -----------------------------------------------------
    taps: list[str] = []
    for i in range(cfg.depth):
        x = _block(c, x, i)
        if i in cfg.intermediate_layers:
            taps.append(x)

    # get_intermediate_layers applies the final norm to each tap, then drops the cls token.
    features: list[str] = []
    for j, tap in enumerate(taps):
        h = c.layernorm(tap, "pretrained.norm", f"tap{j}/norm")
        h = c.emit("slice", [h], f"tap{j}/patches", (1, n_patch, cfg.embed_dim),
                   axis=1, start=1, stop=cfg.num_tokens)
        h = c.transpose(h, (0, 2, 1), f"tap{j}/nchw")
        features.append(c.reshape(h, (1, cfg.embed_dim, grid, grid), f"tap{j}/grid"))

    # -- DPT reassembly --------------------------------------------------
    # resize_layers is [ConvT s4, ConvT s2, Identity, Conv3x3 s2]; index 2 has no weights.
    reassembled: list[str] = []
    for i, feat in enumerate(features):
        h = c.conv(feat, f"depth_head.projects.{i}", f"reassemble{i}/project")
        if i == 0:
            h = c.conv_t(h, "depth_head.resize_layers.0", "reassemble0/up4", stride=4)
        elif i == 1:
            h = c.conv_t(h, "depth_head.resize_layers.1", "reassemble1/up2", stride=2)
        elif i == 3:
            h = c.conv(h, "depth_head.resize_layers.3", "reassemble3/down2", stride=2, pad=1)
        reassembled.append(h)

    rn = [
        c.conv(h, f"depth_head.scratch.layer{i + 1}_rn", f"rn{i + 1}", pad=1, bias=False)
        for i, h in enumerate(reassembled)
    ]

    # -- fusion ladder ---------------------------------------------------
    sizes = [c.shapes[t][2:] for t in rn]
    out = _fusion_block(c, rn[3], None, "depth_head.scratch.refinenet4", "refine4", sizes[2])
    out = _fusion_block(c, out, rn[2], "depth_head.scratch.refinenet3", "refine3", sizes[1])
    out = _fusion_block(c, out, rn[1], "depth_head.scratch.refinenet2", "refine2", sizes[0])
    doubled = (sizes[0][0] * 2, sizes[0][1] * 2)
    out = _fusion_block(c, out, rn[0], "depth_head.scratch.refinenet1", "refine1", doubled)

    # -- output head -----------------------------------------------------
    out = c.conv(out, "depth_head.scratch.output_conv1", "head/conv1", pad=1)
    out = c.interp(out, (grid * cfg.patch_size, grid * cfg.patch_size), "head/up")
    out = c.conv(out, "depth_head.scratch.output_conv2.0", "head/conv2", pad=1)
    out = c.relu(out, "head/relu")
    out = c.conv(out, "depth_head.scratch.output_conv2.2", "head/conv3")
    out = c.relu(out, "depth")

    b.output(out)
    return b.build()

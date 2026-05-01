import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

cfg = Config()


class GoluImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, cfg.num_embed, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.cls = nn.Parameter(torch.randn(1, 1, cfg.num_embed) * 0.02)
        self.box_token = nn.Parameter(torch.randn(1, 1, cfg.num_embed) * 0.02)
        self.blocks = nn.Sequential(*[BlockAttention(cfg.kernel) for _ in range(cfg.num_layer)])
        self.layer_norm = nn.LayerNorm(cfg.num_embed)
        self.class_head = nn.Linear(cfg.num_embed, cfg.num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(cfg.num_embed, cfg.num_embed * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.num_embed * 2, 4),
        )

    def forward(self, x, targets=None):
        x = self.patch(x)
        grid_size = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        batch = x.shape[0]
        x = torch.cat([
            self.cls.expand(batch, -1, -1),
            self.box_token.expand(batch, -1, -1),
            x,
        ], dim=1)
        x = self.blocks((x, grid_size))[0]
        x = self.layer_norm(x)

        class_logits = self.class_head(x[:, 0])
        box = self.box_head(x[:, 1]).sigmoid()
        output = {"class_logits": class_logits, "box": box}

        if targets is None:
            return output, None

        cls_loss = F.cross_entropy(class_logits, targets["label"])
        box_loss = F.smooth_l1_loss(box, targets["box"])
        loss = cfg.cls_loss_weight * cls_loss + cfg.box_loss_weight * box_loss
        losses = {"loss": loss, "cls_loss": cls_loss, "box_loss": box_loss}
        return output, losses

    def print_model_info(self):
        print("\nMODEL INFO")
        total = sum(p.numel() for p in self.parameters())
        print(f"{'total parameters':<20}: \033[1;92m{total}\033[0m")
        cfg.print_config()


class BlockAttention(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.layer_norm = nn.ModuleList([nn.LayerNorm(cfg.num_embed) for _ in range(2)])
        self.attention = MultiHeadPatchAttention(kernel)
        self.feed_forward = FeedForward()
        self.dropout = nn.ModuleList([nn.Dropout(cfg.dropout) for _ in range(2)])

    def forward(self, state):
        x, grid_size = state
        x = x + self.dropout[0](self.attention(self.layer_norm[0](x), grid_size))
        x = x + self.dropout[1](self.feed_forward(self.layer_norm[1](x)))
        return x, grid_size


class MultiHeadPatchAttention(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        self.qkv = nn.Linear(cfg.num_embed, 3 * cfg.head, bias=False)
        self.token_qkv = nn.Linear(cfg.num_embed, 3 * cfg.head, bias=False)
        self.rotation = Rotation()
        self.proj = nn.Linear(cfg.head, cfg.num_embed)
        self.token_proj = nn.Linear(cfg.head, cfg.num_embed)
        self.indices = None
        self.cached_grid_size = None

    def forward(self, x, grid_size):
        special = x[:, :2]
        patches = x[:, 2:]

        batch, tokens, _ = patches.shape
        head_dim = cfg.head // cfg.num_head

        q, k, v = self.qkv(patches).chunk(3, dim=-1)
        q = self.rotation(q.view(batch, tokens, cfg.num_head, head_dim).transpose(1, 2))
        k = self.rotation(k.view(batch, tokens, cfg.num_head, head_dim).transpose(1, 2))
        v = v.view(batch, tokens, cfg.num_head, head_dim).transpose(1, 2)

        indices = self._get_indices(grid_size, patches.device)
        index_shape = indices.shape
        flat_indices = indices.reshape(-1)

        k = torch.index_select(k, 2, flat_indices).view(
            batch, cfg.num_head, index_shape[0], index_shape[1], head_dim
        )
        v = torch.index_select(v, 2, flat_indices).view(
            batch, cfg.num_head, index_shape[0], index_shape[1], head_dim
        )

        patches = self._attention(q.unsqueeze(3), k, v)
        patches = patches.transpose(1, 2).reshape(batch, tokens, cfg.head)
        patches = self.proj(patches)

        q, _, _ = self.token_qkv(special).chunk(3, dim=-1)
        _, k, v = self.token_qkv(patches).chunk(3, dim=-1)
        q = q.view(batch, 2, cfg.num_head, head_dim).transpose(1, 2)
        k = self.rotation(k.view(batch, tokens, cfg.num_head, head_dim).transpose(1, 2))
        v = v.view(batch, tokens, cfg.num_head, head_dim).transpose(1, 2)
        special = self._attention(q.unsqueeze(3), k.unsqueeze(2), v.unsqueeze(2)).squeeze(3)
        special = special.transpose(1, 2).reshape(batch, 2, cfg.head)
        special = self.token_proj(special)

        return torch.cat([special, patches], dim=1)

    def _get_indices(self, grid_size, device):
        if self.indices is not None and self.cached_grid_size == grid_size and self.indices.device == device:
            return self.indices

        height, width = grid_size
        offsets = range(-self.kernel, self.kernel + 1)
        rows = []
        for idx in range(height * width):
            row = idx // width
            col = idx % width
            neighbors = []
            for row_offset in offsets:
                for col_offset in offsets:
                    next_row = min(max(row + row_offset, 0), height - 1)
                    next_col = min(max(col + col_offset, 0), width - 1)
                    neighbors.append(next_row * width + next_col)
            rows.append(neighbors)

        self.indices = torch.tensor(rows, device=device)
        self.cached_grid_size = grid_size
        return self.indices

    def _attention(self, q, k, v):
        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        weights = weights.softmax(dim=-1)
        return weights @ v


class Rotation(nn.Module):
    def __init__(self, base=10000):
        super().__init__()
        head_dim = cfg.head // cfg.num_head
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)

    def forward(self, x):
        tokens = x.shape[-2]
        if self.cos.numel() == 0 or self.cos.shape[-2] < tokens or self.cos.device != x.device:
            freq = torch.outer(torch.arange(tokens, device=x.device).float(), self.inv_freq)
            self.cos = freq.cos()
            self.sin = freq.sin()

        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        cos = self.cos[:tokens].to(dtype=x.dtype)
        sin = self.sin[:tokens].to(dtype=x.dtype)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(cfg.num_embed, cfg.num_embed * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.num_embed * 4, cfg.num_embed),
        )

    def forward(self, x):
        return self.feed_forward(x)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from byte_token.tokenizer import ByteLevelTokenizer

token = ByteLevelTokenizer()
cfg = Config()
state = {}


class Golu(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = nn.Parameter(torch.randn((1, 1, cfg.num_embed)))
        self.linear = nn.Linear(3, cfg.num_embed)
        self.lang_mode_head = nn.Linear(cfg.num_embed, len(token.classes))
        self.blocks = nn.Sequential(
            *[nn.Sequential(*[BlockAttention(cfg.kernel)]) for i in range(cfg.num_layer)])
        self.layer_norm = nn.LayerNorm(cfg.num_embed)

    def forward(self, x, y=None):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        state["height"] = H
        state["width"] = W
        x = x.view(B, H*W, C)
        x = self.linear(x)
        x = torch.cat([self.cls.broadcast_to(B, 1, cfg.num_embed), x], dim=-2)
        x = self.blocks(x)
        x = x[:, 0]
        x = self.layer_norm(x)
        logits = self.lang_mode_head(x)
        if y is None:
            return logits, None
        loss = F.cross_entropy(logits, y)
        return logits, loss

    def print_model_info(self):
        print("\nMODEL INFO")
        print(f"{"total parameters":<20}: \033[1;92m{sum(p.numel()
              for p in self.parameters())}\033[0m")
        cfg.print_config()


class BlockAttention(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(cfg.num_embed) for _ in range(2)])
        self.attention = MultiHeadPatchAttention(
            kernel)
        self.feed_forward = nn.ModuleList([FeedForward(), FeedForward()])
        self.dropout = nn.ModuleList(
            [nn.Dropout(cfg.dropout) for _ in range(2)])

    def forward(self, x):
        x = x + self.dropout[0](self.attention(self.layer_norm[0](x)))
        x = x + self.dropout[1](self.feed_forward[0](self.layer_norm[1](x)))
        return x


class MultiHeadPatchAttention(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        self.qkv = nn.Linear(cfg.num_embed, 3*cfg.head, bias=False)
        self.qkv_cls = nn.Linear(cfg.num_embed, 3*cfg.head, bias=False)
        self.add_rotation = Rotation()
        self.proj = nn.Linear(cfg.head, cfg.num_embed)
        self.proj_cls = nn.Linear(cfg.head, cfg.num_embed)
        self.indices = None

    def forward(self, x: torch.tensor):

        cls = x[:, :1]
        x = x[:, 1:]

        B, T, C = x.shape
        H = cfg.head//cfg.num_head

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.add_rotation(q.view(B, T, cfg.num_head, H).transpose(1, 2))
        k = self.add_rotation(k.view(B, T, cfg.num_head, H).transpose(1, 2))
        v = v.view(B, T, cfg.num_head, H).transpose(1, 2)

        indices = self._get_indices(T)
        i_shape = indices.shape
        indices = indices.view(-1)

        k = torch.index_select(k, 2, indices).view(
            B, cfg.num_head, i_shape[0], i_shape[1], H)
        v = torch.index_select(v, 2, indices).view(
            B, cfg.num_head, i_shape[0], i_shape[1], H)
        q = q.unsqueeze(3)

        x = self._attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, T, cfg.num_head * H)
        x = self.proj(x)

        _, k, v = self.qkv_cls(x).chunk(3, dim=-1)
        q, _, _ = self.qkv_cls(cls).chunk(3, dim=-1)
        q = q.view(B, 1, cfg.num_head, H).transpose(1, 2)
        k = self.add_rotation(k.view(B, T, cfg.num_head, H).transpose(1, 2))
        v = v.view(B, T, cfg.num_head, H).transpose(1, 2)

        cls = self._attention(q, k, v)
        cls = cls.transpose(1, 2).reshape(B, 1, cfg.num_head * H)
        cls = self.proj_cls(cls)
        x = torch.cat([cls, x], dim=1)

        return x

    def _get_indices(self, out):
        height, width = state["height"], state["width"]
        if self.indices is not None and self.height == height and self.width == width:
            return self.indices

        self.height = height
        self.width = width

        rng = range(-self.kernel, self.kernel, 1)

        def filter(i, h, w):
            if (i % width) + w < 0 or (i % width) + w >= width:
                return i
            if (i//height) + h < 0 or (i//height) + h >= height:
                return i
            return i+(h*width)+w

        self.indices = torch.tensor([[filter(i, h, w) for h in rng for w in rng]
                                     for i in range(height*width)])
        return self.indices

    def _attention(self, q, k, v):
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.softmax(-1)
        return wei @ v

    def _expand(self, x, P):
        remainder = x.shape[-2] % P
        if remainder != 0:
            repeat = x[..., [-1], :].expand(x.shape[:-2] + (P-remainder, -1))
            x = torch.cat((x, repeat), dim=-2)

        return x

    def _unfold(self, x, P):
        x = self._expand(x, P)
        T, C = x.shape[-2:]
        return x.view(x.shape[:-2] + (math.ceil(T/P), P, C))

    def _fold(self, x, batch):
        T, P, C = x.shape[-3:]
        return x.reshape(x.shape[:-3] + (T * P, C))[..., :batch, :]


class Rotation(nn.Module):
    def __init__(self, base=10000):
        super().__init__()
        head = cfg.head//cfg.num_head
        inv_freq = 1.0 / base ** (torch.arange(0, head, 2) / head)
        self.register_buffer('inv_freq', inv_freq)
        self.freq = None

    def forward(self, x):
        if self.freq is None or self.freq.shape[-2] < x.shape[-2]:
            self.freq = torch.outer(torch.arange(
                x.shape[-2]).float(), self.inv_freq)
            self.cos = self.freq.cos()
            self.sin = self.freq.sin()

        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([
            x1*self.cos[:x.shape[-2]] - x2*self.sin[:x.shape[-2]],
            x1*self.sin[:x.shape[-2]] + x2*self.cos[:x.shape[-2]]
        ], dim=-1)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(cfg.num_embed, cfg.num_embed * 4),
            nn.GELU(),
            nn.Linear(cfg.num_embed * 4, cfg.num_embed),
            nn.Linear(cfg.num_embed, cfg.num_embed * 4),
            nn.GELU(),
            nn.Linear(cfg.num_embed * 4, cfg.num_embed))

    def forward(self, x):
        return self.feed_forward(x)

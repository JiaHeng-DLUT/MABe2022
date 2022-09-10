import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """
    From: https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L37

    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, opt):
        super().__init__()
        assert opt["hidden_dim"] % opt["num_heads"] == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(opt["hidden_dim"], opt["hidden_dim"])
        self.query = nn.Linear(opt["hidden_dim"], opt["hidden_dim"])
        self.value = nn.Linear(opt["hidden_dim"], opt["hidden_dim"])
        # regularization
        self.attn_drop = nn.Dropout(opt["attn_drop_prob"])
        self.resid_drop = nn.Dropout(opt["res_drop_prob"])
        # output projection
        self.proj = nn.Linear(opt["hidden_dim"], opt["hidden_dim"])
        # causal mask to ensure that attention is only applied to the left in the input sequence
        num_frames = opt["num_frames"]
        num_objects = opt["num_objects"]
        a = torch.tril(torch.ones(num_frames, num_frames))[:, :, None, None]
        b = torch.ones(1, 1, num_objects, num_objects)
        mask = (
            (a * b)
            .transpose(1, 2)
            .reshape(num_frames * num_objects, -1)[None, None, :, :]
        )
        self.register_buffer("mask", mask)
        self.num_heads = opt["num_heads"]

    def forward(self, x, mask, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x)
            .view(B, T, self.num_heads, C // self.num_heads)
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x)
            .view(B, T, self.num_heads, C // self.num_heads)
            .transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill((self.mask * mask[:, None, None, :]) == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """
    From: https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L81

    An unassuming Transformer block.
    """

    def __init__(self, opt):
        super().__init__()
        self.ln1 = nn.LayerNorm(opt["hidden_dim"])
        self.ln2 = nn.LayerNorm(opt["hidden_dim"])
        self.attn = CausalSelfAttention(opt)
        self.mlp = nn.Sequential(
            nn.Linear(opt["hidden_dim"], 4 * opt["hidden_dim"]),
            nn.GELU(),
            nn.Linear(4 * opt["hidden_dim"], opt["hidden_dim"]),
            nn.Dropout(opt["res_drop_prob"]),
        )

    def forward(self, inputs):
        (x, mask) = inputs
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return (x, mask)


class Transformer(nn.Module):
    """
    Attention is only applied to the animals in the left frames.
    """

    def __init__(self, opt):
        super().__init__()

        # input embedding stem
        self.opt = opt
        self.tok_emb = nn.Linear(opt["input_dim"], opt["hidden_dim"])
        self.bn = nn.BatchNorm2d(opt["input_dim"])
        # self.pos_emb = nn.Parameter(torch.zeros(
        #     1, opt['total_frames'], opt['hidden_dim']))
        # transformer
        self.blocks = nn.Sequential(*[Block(opt) for _ in range(opt["num_layers"])])
        # decoder head
        self.ln_f = nn.LayerNorm(opt["hidden_dim"])
        self.proj = nn.Linear(opt["hidden_dim"], opt["output_dim"])

        self.decode_object = opt.get("decode_object", False)
        if self.decode_object:
            self.decoder_animal = nn.Sequential(
                nn.Linear(opt["output_dim"], opt["output_dim"]),
                nn.Tanh(),
                nn.LayerNorm(opt["output_dim"]),
                nn.Linear(opt["output_dim"], opt["input_dim"]),
            )

        self.decode_frame = opt.get("decode_frame", False)
        if self.decode_frame:
            self.decoder_frame = nn.Sequential(
                nn.Linear(opt["output_dim"], opt["output_dim"]),
                nn.Tanh(),
                nn.LayerNorm(opt["output_dim"]),
                nn.Linear(opt["output_dim"], opt["input_dim"] * opt["num_objects"]),
            )

        self.total_frames = opt["total_frames"]
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x, mask):
        (x, mask) = self.blocks((x, mask))
        x = self.ln_f(x)
        x = self.proj(x)  # (b, t * c, output_dim)
        return x

    def forward(
        self, tokens, masks, pos, flip=False, decode_object=False, decode_frame=False
    ):
        """
        Args:
            tokens (torch.Tensor): (b, num_frames, num_objects, input_dim)
            masks (torch.Tensor): (b, num_frames, num_objects)
            pos (torch.Tensor): (b)
            decode_object (bool, optional): whether to decode animals or not.
            decode_frame (bool, optional): whether to decode frames or not.
            flip (bool, optional): whether to flip or not.
        """
        ret = {}

        b, t, c, d = tokens.shape
        # assert t <= self.total_frames, "Cannot forward, pos_emb is exhausted."

        token_embeddings = self.tok_emb(
            self.bn(tokens.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        )
        # position_embeddings = []
        # for i in range(b):
        #     position_embeddings.append(self.pos_emb[:, pos[i]: pos[i] + t, :])
        # position_embeddings = torch.cat(
        #     position_embeddings, dim=0)[:, :, None, :]
        embeddings = token_embeddings.view(b, t * c, -1)
        masks = masks.view(b, -1)

        # LR
        feat_LR = self.encode(embeddings, masks)
        if flip:
            # RL
            feat_RL = self.encode(embeddings.flip(1), masks.flip(1))

        if decode_object:
            masks = masks.view(b, t * c, 1)
            tokens = tokens.view(b, t * c, -1)
            # LR
            animal_LR = (self.decoder_animal(feat_LR) * masks).view(b, t, c, -1)
            gt = (tokens * masks).view(b, t, c, -1)
            animal_LR = animal_LR[:, :-1] - gt[:, 1:]
            ret.update({"animal_LR": animal_LR})
            # RL
            if flip:
                animal_RL = (self.decoder_animal(feat_RL) * masks.flip(1)).view(
                    b, t, c, -1
                )
                gt = (tokens.flip(1) * masks.flip(1)).view(b, t, c, -1)
                animal_RL = animal_RL[:, :-1] - gt[:, 1:]
                ret.update({"animal_RL": animal_RL})

        # LR
        feat_LR = feat_LR.view(b, t, c, -1).mean(dim=-2)
        ret.update({"feat_LR": feat_LR})
        # RL
        if flip:
            feat_RL = feat_RL.view(b, t, c, -1).mean(dim=-2)
            ret.update({"feat_RL": feat_RL})

        if decode_frame:
            # LR
            tokens = tokens.view(b, t, -1)
            frame_LR = self.decoder_frame(feat_LR)
            frame_LR = frame_LR[:, :-1] - tokens[:, 1:]
            ret.update({"frame_LR": frame_LR})
            if flip:
                # RL
                frame_RL = self.decoder_frame(feat_RL)
                frame_RL = frame_RL[:, :-1] - tokens.flip(1)[:, 1:]
                ret.update({"frame_RL": frame_RL})

        return ret

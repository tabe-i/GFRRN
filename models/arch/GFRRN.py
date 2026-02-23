from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
from models.arch.swin_det import swin_large_384_det
from models.arch.GAFLB import FreModule
from models.arch.DAA import DualDynamicAgentBlock
import kornia


def window_partition(x, window_size):
    if not isinstance(window_size, tuple):
        window_size = (window_size, window_size)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    if not isinstance(window_size, tuple):
        window_size = (window_size, window_size)

    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class CABlock(nn.Module):
    def __init__(self, channels):
        super(CABlock, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        return x * self.ca(x)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SinBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block1 = nn.Sequential(
            LayerNorm2d(c),
            nn.Conv2d(c, c * 2, 1),
            nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2),
            SimpleGate(),
            CABlock(c),
            nn.Conv2d(c, c, 1)
        )

        self.block2 = nn.Sequential(
            LayerNorm2d(c),
            nn.Conv2d(c, c * 2, 1),
            SimpleGate(),
            nn.Conv2d(c, c, 1)
        )

        self.a = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.block1(inp)
        x_skip = inp + x * self.a
        x = self.block2(x_skip)
        out = x_skip + x * self.b
        return out


class DualStreamGate(nn.Module):
    def forward(self, x, y):
        x1, x2 = x.chunk(2, dim=1)
        y1, y2 = y.chunk(2, dim=1)
        return x1 * y2, y1 * x2


class DualStreamSeq(nn.Sequential):
    def forward(self, x, y=None):
        y = y if y is not None else x
        for module in self:
            x, y = module(x, y)
        return x, y


class DualStreamBlock(nn.Module):
    def __init__(self, *args):
        super(DualStreamBlock, self).__init__()
        self.seq = nn.Sequential()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.seq.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.seq.add_module(str(idx), module)

    def forward(self, x, y):
        return self.seq(x), self.seq(y)

class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim,hidden_dim,3,1,1,groups=dim),
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0)
        )
        self.act =nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x

class MuGIBlock(nn.Module):
    def __init__(self, c, shared_b=True):
        super().__init__()
        self.block1 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1),
                nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
            ),
            DualStreamGate(),
            DualStreamBlock(CABlock(c)),
            DualStreamBlock(nn.Conv2d(c, c, 1))
        )

        self.a_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.a_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.block2 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1)
            ),
            DualStreamGate(),
            DualStreamBlock(
                nn.Conv2d(c, c, 1)
            )

        )

        self.shared_b = shared_b
        if shared_b:
            self.b = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        else:
            self.b_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
            self.b_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp_l, inp_r):
        x, y = self.block1(inp_l, inp_r)
        x_skip, y_skip = inp_l + x * self.a_l, inp_r + y * self.a_r
        x, y = self.block2(x_skip, y_skip)
        if self.shared_b:
            out_l, out_r = x_skip + x * self.b, y_skip + y * self.b
        else:
            out_l, out_r = x_skip + x * self.b_l, y_skip + y * self.b_r
        return out_l, out_r

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class LayeredWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, num_layers=2, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * 3, num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_l = torch.arange(num_layers)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_l, coords_h, coords_w]))  # 3, Wl, Wh, Ww,
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*2

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wl, Wh*Ww*Wl
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wl, Wh*Ww*Wl, 3
        relative_coords[:, :, 0] += 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[0] - 1
        relative_coords[:, :, 2] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wl, Wh*Ww*Wl
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * 2, self.window_size[0] * self.window_size[1] * 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DualAttentionInteractiveBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, shift_size=0, window_size=12, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        if min(self.input_resolution) <= self.window_size:
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        if min(self.input_resolution) <= self.window_size:
            self.window_size = min(self.input_resolution)
        self.norm1 = DualStreamBlock(norm_layer(dim))
        self.dsia_sa = WindowAttention(dim, to_2tuple(self.window_size), num_heads=num_heads)
        self.dsia_ca = LayeredWindowAttention(dim, to_2tuple(self.window_size), num_heads=num_heads)

        self.feedforward = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(dim),
                nn.Conv2d(dim, dim * 2, 1),
                nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2)
            ),
            DualStreamGate(),
            DualStreamBlock(CABlock(dim)),
            DualStreamBlock(nn.Conv2d(dim, dim, 1))
        )

        self.a = nn.Parameter(torch.zeros((1, 1, dim)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    def create_mask(self, H, W):
        if self.shift_size == 0:
            return None
            
        img_mask = torch.zeros((1, H, W, 1), device='cuda:0')
        
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        base_attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        base_attn_mask = base_attn_mask.masked_fill(base_attn_mask != 0, float(-100.0))
        base_attn_mask = base_attn_mask.masked_fill(base_attn_mask == 0, float(0.0))
        
        win_tokens = self.window_size * self.window_size
        layered_attn_mask = torch.zeros(
            (base_attn_mask.shape[0], win_tokens * 2, win_tokens * 2),
            device=base_attn_mask.device
        )
        
        layered_attn_mask[:, :win_tokens, :win_tokens] = base_attn_mask
        layered_attn_mask[:, win_tokens:, win_tokens:] = base_attn_mask
        layered_attn_mask[:, :win_tokens, win_tokens:] = 0
        layered_attn_mask[:, win_tokens:, :win_tokens] = 0
        
        return base_attn_mask, layered_attn_mask

    def forward(self, x, y):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        y = y.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        x_skip, y_skip = x, y
        x, y = self.norm1(x, y)
        
        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)
    
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x, shifted_y = x, y
        
        if not self.training:
            pad_l = pad_t = 0
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            shifted_x = F.pad(shifted_x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            shifted_y = F.pad(shifted_y, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, Hp, Wp, _ = shifted_x.shape
        else:
            Hp, Wp = H, W
            pad_r, pad_b = 0, 0
        
        base_mask, layered_mask = self.create_mask(Hp, Wp) if self.shift_size > 0 else (None, None)

        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, C)
        y_windows = window_partition(shifted_y, self.window_size).view(-1, self.window_size * self.window_size, C)

        # xx_windows, yy_windows = self.dsia_sa(torch.cat([x_windows, y_windows], dim=0), mask=self.attn_mask).chunk(2, dim=0)
        # xy_windows, yx_windows = self.dsia_ca(torch.cat([x_windows, y_windows], dim=-2), mask=self.attn_mask).chunk(2, dim=-2)
        xx_windows, yy_windows = self.dsia_sa(torch.cat([x_windows, y_windows], dim=0), mask=base_mask).chunk(2, dim=0)
        xy_windows, yx_windows = self.dsia_ca(torch.cat([x_windows, y_windows], dim=-2), mask=layered_mask).chunk(2, dim=-2)

        x_windows = (xx_windows + xy_windows).view(-1, self.window_size, self.window_size, C)
        y_windows = (yy_windows + yx_windows).view(-1, self.window_size, self.window_size, C)

        shifted_x = window_reverse(x_windows, self.window_size, Hp, Wp)  # B H' W' C
        shifted_y = window_reverse(y_windows, self.window_size, Hp, Wp)  # B H' W' C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            y = torch.roll(shifted_y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x, y = shifted_x, shifted_y

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
            y = y[:, :H, :W, :].contiguous()

        x = x_skip + x.view(B, H * W, C) * self.a
        y = y_skip + y.view(B, H * W, C) * self.a

        x_skip = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        y_skip = y.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x, y = self.feedforward(x_skip, y_skip)
        x, y = x_skip + x * self.b, y_skip + y * self.b
        return x, y

class LocalFeatureExtractor(nn.Module):
    def __init__(self, dims, enc_blk_nums=[]):
        super(LocalFeatureExtractor, self).__init__()
        self.dims = dims
        c = dims
        self.stem = DualStreamBlock(nn.Conv2d(3, c, 3, padding=1))
        self.block1 = DualStreamSeq(
            *[MuGIBlock(c) for _ in range(enc_blk_nums[0])],
            DualStreamBlock(nn.Conv2d(c, c * 2, 2, 2))
        )
        c *= 2

        self.block2 = DualStreamSeq(
            *[MuGIBlock(c) for _ in range(enc_blk_nums[1])],
            DualStreamBlock(nn.Conv2d(c, c * 2, 2, 2))
        )

        c *= 2
        self.block3 = DualStreamSeq(
            *[MuGIBlock(c) for _ in range(enc_blk_nums[2])],
            DualStreamBlock(nn.Conv2d(c, c * 2, 2, 2))
        )
        
        c *= 2
        self.block4 = DualStreamSeq(
            *[MuGIBlock(c) for _ in range(enc_blk_nums[3])],
            DualStreamBlock(nn.Conv2d(c, c * 2, 2, 2))
        )

        c *= 2
        self.block5 = DualStreamSeq(
            *[MuGIBlock(c) for _ in range(enc_blk_nums[4])],
            DualStreamBlock(nn.Conv2d(c, c * 2, 2, 2))
        )

    def forward(self, x):
        x0, y0 = self.stem(x, x)
        x1, y1 = self.block1(x0, y0)
        x2, y2 = self.block2(x1, y1)
        x3, y3 = self.block3(x2, y2)
        x4, y4 = self.block4(x3, y3)
        x5, y5 = self.block5(x4, y4)
        return (x0, y0), (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)



def check_mona_modes_automatically(model):

    print("=== Mona mode automatic check ===")

    mona_modules = []
    other_modules = []
    
    for name, module in model.named_modules():
        if 'my_module' in name:
            mona_modules.append((name, module.training))
        else:
            other_modules.append((name, module.training))
    
    print("Mona Block:")
    correct_mona_count = 0
    for name, training in mona_modules:
        status = "training" if training else "evaluation"
        if training:  # Mona层应该处于训练模式
            # print(f"  ✓ {name}: {status}")
            correct_mona_count += 1
        else:
            print(f"  ✗ {name}: {status} error (should be in training mode)")
    
    print("\nOther Layers:")
    correct_other_count = 0
    for name, training in other_modules:
        status = "training" if training else "evaluation"
        if not training:  # 其他层应该处于评估模式
            # print(f"  ✓ {name}: {status}")
            correct_other_count += 1
        else:
            print(f"  ✗ {name}: {status} error (should be in evaluation mode)")

    
    # 计算正确率
    total_modules = len(mona_modules) + len(other_modules)
    correct_modules = correct_mona_count + correct_other_count
    accuracy = correct_modules / total_modules * 100 if total_modules > 0 else 0
    
    print(f"accuracy: {accuracy:.2f}%")
    
    return {
        'mona_modules': mona_modules,
        'other_modules': other_modules,
        'mona_accuracy': correct_mona_count / len(mona_modules) * 100 if mona_modules else 0,
        'other_accuracy': correct_other_count / len(other_modules) * 100 if other_modules else 0,
        'total_accuracy': accuracy
    }



class GFRRN(nn.Module):
    def __init__(self, args, input_resolution=(384, 384), window_size=12, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        if args is None:
            self.swin_prior = swin_large_384_det('weights/swin_large_o365_finetune.pth')
        else:
            self.swin_prior = swin_large_384_det(args.backbone_weight_path)
        self.swin_prior.eval()

        for param in self.swin_prior.parameters():
            param.requires_grad = False

        for name, param in self.swin_prior.named_parameters():
            if 'my_module' in name:
                param.requires_grad = True

    
        def set_mona_train_mode(module):
            module_name = str(module.__class__.__name__)
            if 'Mona' in module_name:
                module.train()  
         
        self.swin_prior.apply(set_mona_train_mode)

        # check_mona_modes_automatically(self.swin_prior)
    
        self.conv_prior = LocalFeatureExtractor(48, [2, 2, 2, 2, 2])

        self.input_resolution = input_resolution
        self.device = 'cuda'
        self.window_size = window_size
        H, W = input_resolution

        self.aib5 = DualStreamSeq(
            DualStreamBlock(nn.PixelShuffle(2)),
            DualAttentionInteractiveBlock(384, (H // 16, W // 16), 8, window_size=window_size),
            DualStreamBlock(nn.Conv2d(in_channels=384, out_channels=768, kernel_size=1)),

        )

        self.aib4 = DualStreamSeq(
            DualAttentionInteractiveBlock(768, (H // 16, W // 16), 8, window_size=window_size),
        )
        self.fre4 = FreModule(768, 8, False)
        self.lib4 = DualStreamSeq(
            *[DualAttentionInteractiveBlock(768, (H // 16, W // 16), 8, window_size=window_size) for i in range(enc_blk_nums[0])],
            DualStreamBlock(nn.PixelShuffle(2)),
            *[MuGIBlock(192) for _ in range(dec_blk_nums[0])],
            DualStreamBlock(nn.Conv2d(in_channels=192, out_channels=384, kernel_size=1))
        )
        

        self.aib3 = DualStreamSeq(
            DualAttentionInteractiveBlock(384, (H // 8, W // 8), 8, window_size=window_size),
        )

        self.fre3 = FreModule(384, 8, False)
        self.lib3 = DualStreamSeq(
            *[DualAttentionInteractiveBlock(384, (H // 8, W // 8), 8, window_size=window_size) for i in range(enc_blk_nums[1])],
            DualStreamBlock(nn.PixelShuffle(2)),
            *[MuGIBlock(96) for _ in range(dec_blk_nums[1])],
            DualStreamBlock(nn.Conv2d(in_channels=96, out_channels=192, kernel_size=1))
        )
        

        self.aib2 = DualStreamSeq(
            DualAttentionInteractiveBlock(192, (H // 4, W // 4), 4, window_size=window_size),
        )

        self.fre2 = FreModule(192, 4, False)
        self.lib2 = DualStreamSeq(
            *[DualDynamicAgentBlock(192, (H // 4, W // 4), 4, window_size=window_size, max_agent_num=64, min_agent_num=36) for i in range(enc_blk_nums[2])],
            DualStreamBlock(nn.PixelShuffle(2)),
            *[MuGIBlock(48) for _ in range(dec_blk_nums[2])],
            DualStreamBlock(nn.Conv2d(in_channels=48, out_channels=96, kernel_size=1))
        )
        

        self.fre1 = FreModule(96, 2, False)
        self.lib1 = DualStreamSeq(
            *[DualDynamicAgentBlock(96, (H // 2, W // 2), 2, window_size=window_size, max_agent_num=49, min_agent_num=25) for i in range(enc_blk_nums[3])],
            DualStreamBlock(nn.PixelShuffle(2)),
            *[MuGIBlock(24) for _ in range(dec_blk_nums[3])],
            DualStreamBlock(nn.Conv2d(in_channels=24, out_channels=48, kernel_size=1))
        )
        self.fre0 = FreModule(48, 1, False)
        self.lib0 = DualStreamSeq(
            *[DualDynamicAgentBlock(48, (H, W), 1, window_size=window_size, max_agent_num=49, min_agent_num=25) for i in range(enc_blk_nums[4])],
            *[MuGIBlock(48) for _ in range(dec_blk_nums[4])],
        )

        self.out = DualStreamBlock(nn.Conv2d(in_channels=48, out_channels=3, kernel_size=3, padding=1))
        self.lrm = nn.Sequential(
            SinBlock(48),
            nn.Conv2d(48, 3, 3, padding=1),
            nn.Tanh()
        )

    def train(self, mode=True):
        super().train(mode)
        self.swin_prior.eval()

    def forward(self, inp, fn=None):
        inp_ycbcr = kornia.color.rgb_to_ycbcr(inp)
        sp2, sp3, sp4, sp5 = self.swin_prior(inp)

        (cp0_l, cp0_r), (cp1_l, cp1_r), (cp2_l, cp2_r), (cp3_l, cp3_r), (cp4_l, cp4_r), (cp5_l, cp5_r) = self.conv_prior(inp_ycbcr)

        scp5_l, csp5_l = self.aib5(sp5, cp5_l)
        scp5_r, csp5_r = self.aib5(sp5, cp5_r)

        scp4_l, csp4_l = self.aib4(sp4, cp4_l)
        scp4_r, csp4_r = self.aib4(sp4, cp4_r)

        f4_l = self.fre4(inp, scp4_l + csp4_l + scp5_l + csp5_l)
        f4_r = self.fre4(inp, scp4_r + csp4_r + scp5_r + csp5_r)
        f4_l, f4_r = self.lib4(f4_l, f4_r)

        scp3_l, csp3_l = self.aib3(sp3, cp3_l)
        scp3_r, csp3_r = self.aib3(sp3, cp3_r)

        f3_l = self.fre3(inp, f4_l + scp3_l + csp3_l)
        f3_r = self.fre3(inp, f4_r + scp3_r + csp3_r)
        f3_l, f3_r = self.lib3(f3_l, f3_r)

        scp2_l, csp2_l = self.aib2(sp2, cp2_l)
        scp2_r, csp2_r = self.aib2(sp2, cp2_r)

        f2_l = self.fre2(inp, f3_l + scp2_l + csp2_l)
        f2_r = self.fre2(inp, f3_r + scp2_r + csp2_r)
        f2_l, f2_r = self.lib2(f2_l, f2_r)
        f1_l = self.fre1(inp, f2_l + cp1_l)
        f1_r = self.fre1(inp, f2_r + cp1_r)
        f1_l, f1_r = self.lib1(f1_l, f1_r)
        f0_l = self.fre0(inp, f1_l + cp0_l)
        f0_r = self.fre0(inp, f1_r + cp0_r)
        f0_l, f0_r = self.lib0(f0_l, f0_r)
        out_l, out_r = self.out(f0_l, f0_r)
        out_rr = self.lrm(f0_l + f0_r)
        return out_l, out_r, out_rr

if __name__ == '__main__':
    inp = torch.randn(1, 3, 384, 384).cuda()
    model = GFRRN(args = None,enc_blk_nums=[1, 1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1, 1])
    model = model.cuda()
    print(model(inp)[0].shape)
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import to_2tuple, trunc_normal_
from collections import OrderedDict


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


class DynamicAgentAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 max_agent_num=64, window=12, min_agent_num=36, eps=1e-6, version=0, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.max_agent_num = max_agent_num
        self.min_agent_num = min_agent_num
        self.window = window
        self.eps = eps
        self.version = version

        # QKV & output projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Depthwise conv enhance
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim)

        pool_size = int(math.sqrt(max_agent_num))
        assert pool_size * pool_size == max_agent_num, "max_agent_num must be a perfect square"
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self._pool_size = pool_size
        self.A = max_agent_num

        self.complexity_predictor = nn.Sequential(
            nn.Conv2d(dim, max(dim // 8, 1), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(max(dim // 8, 1), 1),
            nn.Sigmoid()
        )

        self.agent_importance_predictor = nn.Sequential(
            nn.Linear(dim, max(dim // 4, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(dim // 4, 1), 1)
        )

        self.agent_token_bias = nn.Parameter(torch.zeros(num_heads, self.A, window * window))
        self.token_agent_bias = nn.Parameter(torch.zeros(num_heads, window * window, self.A))

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.agent_token_bias, std=.02)
        trunc_normal_(self.token_agent_bias, std=.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        assert H * W == N, "N must be perfect square"

        num_heads = self.num_heads
        head_dim = C // num_heads
        A = self.A

        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, N, C)

        q_t = q.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        agent_map = self.pool(q_t)  # (B, C, pool_size, pool_size)
        agent_tokens = agent_map.reshape(B, C, A).permute(0, 2, 1)  # (B, A, C)

        complexity_score = self.complexity_predictor(q_t)  # (B, 1), in [0,1]
        dynamic_count = (self.min_agent_num + (self.max_agent_num - self.min_agent_num) * complexity_score)
        dynamic_count = dynamic_count.squeeze(-1)  # (B,)

        if self.version == 0:
            agent_logits = self.agent_importance_predictor(agent_tokens).squeeze(-1)  # (B, A)
            importance = torch.sigmoid(agent_logits)  # (B, A), in (0,1)
            sum_imp = importance.sum(dim=1, keepdim=True)  # (B,1)
            sum_imp = sum_imp + self.eps
            weights = importance / sum_imp * dynamic_count.unsqueeze(1)  # (B, A) - continuous gating weights
        else:
            weights = (dynamic_count / self.A).unsqueeze(1).expand(-1, self.A)  # (B, A) - uniform weights

        masked_agent_tokens = agent_tokens * weights.unsqueeze(-1)  # (B, A, C)

        q = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)  # (B, H, N, D)
        k = k.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)  # (B, H, N, D)
        v = v.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)  # (B, H, N, D)
        agent_tokens_h = masked_agent_tokens.reshape(B, A, num_heads, head_dim).permute(0, 2, 1, 3)  # (B, H, A, D)


        # agent_token_bias: (num_heads, A, N) -> expand -> (B, num_heads, A, N)
        position_bias = self.agent_token_bias.unsqueeze(0).expand(B, -1, -1, -1).clone()  # (B, H, A, N)
        position_bias = position_bias * weights.view(B, 1, A, 1)

        attn_logits = (agent_tokens_h * self.scale) @ k.transpose(-2, -1)  # (B, H, A, N)
        attn = F.softmax(attn_logits + position_bias, dim=-1)
        attn = self.attn_drop(attn)
        agent_v = attn @ v  # (B, H, A, D)

        # token_agent_bias: (num_heads, N, A) -> expand -> (B, H, N, A)
        agent_bias = self.token_agent_bias.unsqueeze(0).expand(B, -1, -1, -1).clone()  # (B, H, N, A)
        # apply weights along agent dim
        agent_bias = agent_bias * weights.view(B, 1, 1, A)

        q_attn_logits = (q * self.scale) @ agent_tokens_h.transpose(-2, -1)  # (B, H, N, A)
        q_attn = F.softmax(q_attn_logits + agent_bias, dim=-1)
        q_attn = self.attn_drop(q_attn)
        x_attn = q_attn @ agent_v  # (B, H, N, D)

        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        v_reshaped = v.transpose(1, 2).reshape(B, N, C).reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        dwc_feat = self.dwc(v_reshaped)  # (B, C, H, W)
        dwc_feat = dwc_feat.permute(0, 2, 3, 1).reshape(B, N, C)  # (B, N, C)
        x_enhanced = x_attn + dwc_feat

        x_out = self.proj(x_enhanced)
        x_out = self.proj_drop(x_out)

        return x_out

class LayeredDynamicAgentAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, num_layers=2, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0., max_agent_num=64, min_agent_num=36, eps=1e-6, version=0, **kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.version = version
        
        assert int(math.sqrt(min_agent_num)) ** 2 == min_agent_num, "min_agent_num must be a perfect square"
        assert int(math.sqrt(max_agent_num)) ** 2 == max_agent_num, "max_agent_num must be a perfect square"
        self.max_agent_num = max_agent_num
        self.min_agent_num = min_agent_num
        self.eps = eps

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim)

        pool_size = int(math.sqrt(max_agent_num))
        assert pool_size * pool_size == max_agent_num, "max_agent_num must be a perfect square"
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self._pool_size = pool_size
        self.A = max_agent_num

        self.complexity_predictor = nn.Sequential(
            nn.Conv2d(dim, dim // 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim // 8, 1),
            nn.Sigmoid()
        )

        self.agent_importance_predictor = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, 1)
        )

        window_token_count = window_size[0] * window_size[1] * num_layers
        total_agent_count = max_agent_num * num_layers
        
        self.agent_token_bias = nn.Parameter(torch.zeros(num_heads, total_agent_count, window_token_count))
        self.token_agent_bias = nn.Parameter(torch.zeros(num_heads, window_token_count, total_agent_count))
        
        trunc_normal_(self.agent_token_bias, std=.02)
        trunc_normal_(self.token_agent_bias, std=.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

        self.softmax = nn.Softmax(dim=-1)

    def _init_weights(self):
        trunc_normal_(self.agent_token_bias, std=.02)
        trunc_normal_(self.token_agent_bias, std=.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        window_area = self.window_size[0] * self.window_size[1]
        assert N == window_area * self.num_layers, "N must equal window_area * num_layers"
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        H = W = int(math.sqrt(window_area))
        assert H * W == window_area, "window_area must be a perfect square"
        
        q_reshaped = q.transpose(1, 2).reshape(B_, N, C)
        q_t = q_reshaped.reshape(B_, self.num_layers, H, W, C).permute(0, 1, 4, 2, 3)

        agent_tokens_per_layer = []
        complexity_scores = []
        
        for l in range(self.num_layers):
            q_layer = q_t[:, l, :, :, :]
            
            agent_map = self.pool(q_layer) 
            agent_tokens = agent_map.reshape(B_, C, self.max_agent_num).permute(0, 2, 1)
            agent_tokens_per_layer.append(agent_tokens)

            complexity_score = self.complexity_predictor(q_layer)
            complexity_scores.append(complexity_score)

        agent_tokens = torch.cat(agent_tokens_per_layer, dim=1)
        total_agent_count = self.num_layers * self.max_agent_num
        
        complexity_scores = torch.stack(complexity_scores, dim=1)
        complexity_score = torch.mean(complexity_scores, dim=1)

        dynamic_count = (self.min_agent_num + 
                        (self.max_agent_num - self.min_agent_num) * complexity_score)
        dynamic_count = dynamic_count.squeeze(-1)

        agent_logits = self.agent_importance_predictor(agent_tokens)
        agent_logits = agent_logits.squeeze(-1)

        importance = torch.sigmoid(agent_logits)

        if self.version == 0:
            sum_imp = importance.sum(dim=1, keepdim=True)
            sum_imp = sum_imp + self.eps
            weights = importance / sum_imp * dynamic_count.unsqueeze(1)
        else:
            weights = (dynamic_count / total_agent_count).unsqueeze(1).expand(-1, total_agent_count)  # uniform weights

        masked_agent_tokens = agent_tokens * weights.unsqueeze(-1)

        agent_tokens_h = masked_agent_tokens.reshape(B_, total_agent_count, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        position_bias = self.agent_token_bias.unsqueeze(0).expand(B_, -1, -1, -1).clone()
        position_bias = position_bias * weights.view(B_, 1, total_agent_count, 1)
        
        attn_logits = (agent_tokens_h * self.scale) @ k.transpose(-2, -1)
        attn_agent = F.softmax(attn_logits + position_bias, dim=-1)
        attn_agent = self.attn_drop(attn_agent)
        agent_v = attn_agent @ v

        agent_bias = self.token_agent_bias.unsqueeze(0).expand(B_, -1, -1, -1).clone()
        agent_bias = agent_bias * weights.view(B_, 1, 1, total_agent_count)
        
        q_attn_logits = (q * self.scale) @ agent_tokens_h.transpose(-2, -1)
        q_attn = F.softmax(q_attn_logits + agent_bias, dim=-1)
        q_attn = self.attn_drop(q_attn)
        x_agent = q_attn @ agent_v 

        x_agent = x_agent.transpose(1, 2).reshape(B_, N, C)

        v_reshaped = v.transpose(1, 2).reshape(B_, N, C)
        v_per_layer = v_reshaped.reshape(B_, self.num_layers, H, W, C).permute(0, 1, 4, 2, 3)
        
        enhanced_layers = []
        for l in range(self.num_layers):
            v_layer = v_per_layer[:, l, :, :, :]
            enhanced = self.dwc(v_layer)
            enhanced_layers.append(enhanced.permute(0, 2, 3, 1).reshape(B_, H*W, C))
        
        dwc_feat = torch.cat(enhanced_layers, dim=1)
        
        x_combined = x_agent + dwc_feat

        x_out = self.proj(x_combined)
        x_out = self.proj_drop(x_out)
        
        return x_out

    
class DualDynamicAgentBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=12, norm_layer=nn.LayerNorm, max_agent_num=64, min_agent_num=36):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size

        if min(self.input_resolution) <= self.window_size:
            self.window_size = min(self.input_resolution)

        self.norm1 = DualStreamBlock(norm_layer(dim))
        self.dsia_sa = DynamicAgentAttention(dim, num_heads=num_heads,max_agent_num=max_agent_num, min_agent_num=min_agent_num, window=window_size)
        self.dsia_ca = LayeredDynamicAgentAttention(dim, num_heads=num_heads, max_agent_num=max_agent_num, min_agent_num=min_agent_num, window_size=to_2tuple(window_size))

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

    def forward(self, x, y):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        y = y.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        x_skip, y_skip = x, y
        x, y = self.norm1(x, y)
        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)
        
        if not self.training:
            pad_l = pad_t = 0
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            y = F.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, Hp, Wp, _ = x.shape
        else:
            Hp, Wp = H, W
            pad_r, pad_b = 0, 0

        x_windows = window_partition(x, self.window_size).view(-1, self.window_size * self.window_size, C)
        y_windows = window_partition(y, self.window_size).view(-1, self.window_size * self.window_size, C)

        xx_windows, yy_windows = self.dsia_sa(torch.cat([x_windows, y_windows], dim=0)).chunk(2, dim=0)
        xy_windows, yx_windows = self.dsia_ca(torch.cat([x_windows, y_windows], dim=-2)).chunk(2, dim=-2)

        x_windows = (xx_windows + xy_windows).view(-1, self.window_size, self.window_size, C)
        y_windows = (yy_windows + yx_windows).view(-1, self.window_size, self.window_size, C)

        x = window_reverse(x_windows, self.window_size, Hp, Wp)  # B H' W' C
        y = window_reverse(y_windows, self.window_size, Hp, Wp)  # B H' W' C

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
    
if __name__ == "__main__":
    import torch

    B, C, H, W = 2, 96, 24, 24
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, C, H, W)

    model = DualDynamicAgentBlock(dim=C, input_resolution=(H, W), num_heads=8, window_size=12)
    out_x, out_y = model(x, y)
    print(out_x.shape)  # Expected: (B, C, H, W)
    print(out_y.shape)  # Expected: (B, C, H, W)
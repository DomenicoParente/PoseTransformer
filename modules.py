import numpy as np
import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class TubeletEmbedding(nn.Module):
    def __init__(self, p_t: int, p_w: int, p_h: int, dim: int, t_dim: int):
        super(TubeletEmbedding, self).__init__()
        self.tubelet_net = nn.Sequential(
            Rearrange('b c (t pt) (h ph) (w pw) -> b t (h w) (pt ph pw c)', pt=p_t, pw=p_w, ph=p_h),
            nn.Linear(t_dim, dim)
        )

    def forward(self, x):
        return self.tubelet_net(x)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP(nn.Module):
    def __init__(self, dim: int, hdim: int, dropout=0.):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hdim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hdim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MLPhead(nn.Module):
    def __init__(self, dim: int, dim_out: int):
        super(MLPhead, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class PosLinear(nn.Module):
    """Final linear layer that gives the position as result"""

    def __init__(self, dim: int, outdim=3):
        super(PosLinear, self).__init__()
        self.linear = nn.Linear(dim, outdim, bias=True)
        self.initialize_weights()

    def forward(self, x):
        x = self.linear(x)
        return x

    def initialize_weights(self):
        nn.init.kaiming_uniform_(self.linear.weight.data)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias.data, 0)



class OriLinear(nn.Module):
    """Final linear layer that gives the orientation as quaternion"""

    def __init__(self, dim: int, outdim=4):
        super(OriLinear, self).__init__()
        self.linear = nn.Linear(dim, outdim, bias=True)
        self.initialize_weights()

    def forward(self, x):
        x = self.linear(x)
        return x

    def initialize_weights(self):
        nn.init.kaiming_uniform_(self.linear.weight.data)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias.data, 0)


class AttentionFirst(nn.Module):
    def __init__(self, dim, heads=8, dropout=0., dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    """Transformer Encoder as described in paper "ViViT: A Video Vision Transformer" """

    def __init__(self, dim: int, heads: int, mlp_dim: int, depth_l: int, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.attention = AttentionFirst(dim, heads, dropout)
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth_l):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, self.attention),
                PreNorm(dim, self.attention),
                PreNorm(dim, MLP(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        b = x.shape[0]
        x = torch.flatten(x, start_dim=0, end_dim=1)  # extract spatial tokens from x

        for sp_attn, temp_attn, mlp in self.layers:
            sp_x = sp_attn(x) + x  # Spatial attention

            # Reshape tensors for temporal attention
            sp_x = sp_x.chunk(b, dim=0)
            sp_x = [temp[None] for temp in sp_x]
            sp_x = torch.cat(sp_x, dim=0).transpose(1, 2)
            sp_x = torch.flatten(sp_x, start_dim=0, end_dim=1)

            temp_x = temp_attn(sp_x) + sp_x  # Temporal attention

            x = mlp(temp_x) + temp_x  # MLP

            # Again reshape tensor for spatial attention
            x = x.chunk(b, dim=0)
            x = [temp[None] for temp in x]
            x = torch.cat(x, dim=0).transpose(1, 2)
            x = torch.flatten(x, start_dim=0, end_dim=1)

        # Reshape vector to [b, nt, nh*nw, dim]
        x = x.chunk(b, dim=0)
        x = [temp[None] for temp in x]
        x = torch.cat(x, dim=0)
        x = torch.flatten(x, start_dim=2, end_dim=3)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class DropPath(nn.Module):

    def __init__(self, dropout_p=None):
        super(DropPath, self).__init__()
        self.dropout_p = dropout_p

    def forward(self, x):
        return self.drop_path(x, self.dropout_p, self.training)

    def drop_path(self, x, dropout_p=0., training=False):
        if dropout_p == 0. or not training:
            return x
        keep_prob = 1 - dropout_p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape).type_as(x)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return


class PatchEmbed(nn.Module):

    def __init__(self, width, height, p_w, p_h, p_t, channels=3, embed_dims=768):
        super().__init__()
        self.width = width
        self.height = height
        self.patch_width = p_w
        self.patch_height = p_h
        self.patch_time = p_t

        num_patches = \
            (self.width // self.patch_width) * \
            (self.height // self.patch_height)
        self.num_patches = num_patches

        # Use convolutional layer to embed
        self.projection = nn.Conv3d(channels, embed_dims,
                                    kernel_size=(self.patch_time, self.patch_height, self.patch_width),
                                    stride=(self.patch_time, self.patch_height, self.patch_width))

    def forward(self, x):
        x = self.projection(x)
        x = rearrange(x, 'b c t h w -> (b t) (h w) c')
        return x


class MultiheadAttentionWithPreNorm(nn.Module):

    def __init__(self, embed_dims, num_heads, attn_drop=0., proj_drop=0., dropout=0., norm_layer=nn.LayerNorm):
        super(MultiheadAttentionWithPreNorm, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.norm = norm_layer(embed_dims)
        self.attn = Attention(embed_dims, num_heads, qkv_bias=True, attn_drop=attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop = DropPath(proj_drop)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        attn_out, attn_weights = self.attn(x)
        x = residual + self.drop(attn_out)

        return x


class FFNWithPreNorm(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 hidden_channels=1024,
                 num_layers=2,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 dropout_p=0.):
        super().__init__()
        assert num_layers >= 2, 'num_layers should be no less ' \
                                f'than 2. got {num_layers}.'
        self.embed_dims = embed_dims
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.drop = DropPath(dropout_p)
        self.norm = norm_layer(embed_dims)
        layers = []
        in_channels = embed_dims
        for _ in range(num_layers - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, hidden_channels),
                    act_layer(),
                    nn.Dropout(dropout_p)))
            in_channels = hidden_channels
        layers.append(nn.Linear(hidden_channels, embed_dims))
        layers.append(nn.Dropout(dropout_p))
        self.layers = nn.ModuleList(layers)

        self.layer_drop = DropPath(dropout_p)

    def forward(self, x):
        residual = x

        x = self.norm(x)
        for layer in self.layers:
            x = layer(x)

        return residual + self.layer_drop(x)


class TransformerContainer(nn.Module):
    """This class concatenate the series of transformers """

    def __init__(self,
                 num_transformer_layers,
                 embed_dims,
                 num_heads,
                 num_frames,
                 hidden_channels,
                 operator_order,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.num_transformer_layers = num_transformer_layers

        dpr = np.linspace(0, drop_path_rate, num_transformer_layers)
        for i in range(num_transformer_layers):
            self.layers.append(
                BasicTransformerBlock(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    num_frames=num_frames,
                    hidden_channels=hidden_channels,
                    operator_order=operator_order,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    num_layers=num_layers,
                    dpr=dpr[i]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BasicTransformerBlock(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_frames,
                 hidden_channels,
                 operator_order,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 num_layers=2,
                 dpr=0,
                 ):

        super().__init__()
        self.attentions = nn.ModuleList([])
        self.ffns = nn.ModuleList([])

        for i, operator in enumerate(operator_order):
            if operator == 'self_attn':
                self.attentions.append(
                    MultiheadAttentionWithPreNorm(
                        embed_dims=embed_dims,
                        num_heads=num_heads,
                        norm_layer=nn.LayerNorm))

            elif operator == 'ffn':
                self.ffns.append(
                    FFNWithPreNorm(
                        embed_dims=embed_dims,
                        hidden_channels=hidden_channels,
                        num_layers=num_layers,
                        act_layer=act_layer,
                        norm_layer=norm_layer))

    def forward(self, x):

        for layer in self.attentions:
            x = layer(x)
        for layer in self.ffns:
            x = layer(x)
        return x


class PoseLoss(nn.Module):
    def __init__(self, device, sx=0.0, sq=0.0, learn_beta=False):
        super(PoseLoss, self).__init__()
        self.learn_beta = learn_beta

        if not self.learn_beta:
            self.sx = 0
            self.sq = -6.25

        self.sx = nn.Parameter(torch.Tensor([sx]).to(device), requires_grad=self.learn_beta)
        self.sq = nn.Parameter(torch.Tensor([sq]).to(device), requires_grad=self.learn_beta)


        self.loss_print = None

    def forward(self, pred_x, pred_q, target_x, target_q):
        pred_q = F.normalize(pred_q, p=2, dim=1)
        loss_x = F.l1_loss(pred_x, target_x)
        loss_q = F.l1_loss(pred_q, target_q)

        loss = torch.exp(-self.sx) * loss_x \
               + self.sx \
               + torch.exp(-self.sq) * loss_q \
               + self.sq

        self.loss_print = [loss.item(), loss_x.item(), loss_q.item()]

        return loss, loss_x.item(), loss_q.item()

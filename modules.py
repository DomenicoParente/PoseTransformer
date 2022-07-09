import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F


class TubeletEmbed(nn.Module):

    def __init__(self, time, p_w, p_h, p_t, channels=3, embed_dims=768):
        super().__init__()
        self.patch_width = p_w
        self.patch_height = p_h
        self.patch_time = p_t
        self.batch_norm = nn.BatchNorm3d(time)

        # Use convolutional layer to embed
        self.projection = nn.Conv3d(channels, embed_dims,
                                    kernel_size=(self.patch_time, self.patch_height, self.patch_width),
                                    stride=(self.patch_time, self.patch_height, self.patch_width))
        self.fc = nn.Linear(embed_dims, embed_dims)

        self.initialize_weights(self.projection)

    def initialize_weights(self, module):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.trunc_normal_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.batch_norm(x)
        x = rearrange(x, 'b t c h w -> b c t h w')
        x = self.projection(x)
        x = rearrange(x, 'b c t h w -> (b t) (h w) c')
        x = self.fc(x)
        return x

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

        # Use convolutional 3d layer to embed video tubelet
        self.projection = nn.Conv3d(channels, embed_dims,
                                    kernel_size=(self.patch_time, self.patch_height, self.patch_width),
                                    stride=(self.patch_time, self.patch_height, self.patch_width))
        self.init_weights(self.projection)

    def init_weights(self, module):
        nn.init.trunc_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)

    def forward(self, x):
        x = self.projection(x)
        x = rearrange(x, 'b c t h w -> (b t) (h w) c')
        return x


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


class MLP_head(nn.Module):
    def __init__(self, dim: int, dim_out: int):
        super(MLP_head, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_out)
        )

        self.init_weights()

    def forward(self, x):
        return self.net(x)

    def init_weights(self):
        mod = self.net
        for m in mod:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)


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
        return output


class MultiheadAttentionWithPreNorm(nn.Module):
    def __init__(self, embed_dims, num_heads, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.norm = norm_layer(embed_dims)
        self.attn = Attention(embed_dims, num_heads, qkv_bias=True, attn_drop=attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.id = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.norm(x)
        attn_out, attn_weights = self.attn(x)
        x = residual + self.id(attn_out)

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
        self.id = nn.Identity()

    def forward(self, x):
        residual = x

        x = self.norm(x)
        for layer in self.layers:
            x = layer(x)

        return residual + self.id(x)


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
                    num_layers=num_layers))

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


class PosLinear(nn.Module):
    """Final linear layer that gives the position as result"""

    def __init__(self, t: int, dim: int, outdim=3, dropout=0.0):
        super(PosLinear, self).__init__()
        self.fc1 = nn.Linear(dim, outdim, bias=True)
        self.batch_norm = nn.BatchNorm1d(t)
        self.dropout = nn.Dropout(dropout)
        self.initialize_weights()

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return x

    def initialize_weights(self):
        nn.init.trunc_normal_(self.fc1.weight.data)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias.data, 0)


class OriLinear(nn.Module):
    """Final linear layer that gives the orientation as quaternion"""

    def __init__(self, t: int, dim: int, outdim=4, dropout=0.0):
        super(OriLinear, self).__init__()
        self.fc1 = nn.Linear(dim, outdim, bias=True)
        self.batch_norm = nn.BatchNorm1d(t)
        self.dropout = nn.Dropout(dropout)
        self.initialize_weights()

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return x

    def initialize_weights(self):
        nn.init.trunc_normal_(self.fc1.weight.data)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias.data, 0)


class PoseLoss(nn.Module):
    def __init__(self, device, lossf, sq=-6.25, sx=0.0, learn_beta=True):
        super(PoseLoss, self).__init__()
        self.learn_beta = learn_beta
        self.sx = nn.Parameter(torch.Tensor([sx]).to(device), requires_grad=self.learn_beta)
        self.sq = nn.Parameter(torch.Tensor([sq]).to(device), requires_grad=self.learn_beta)
        self.loss_print = None
        # Loss function choice
        if lossf == "smooth":
            self.loss_fun = torch.nn.SmoothL1Loss(beta=0.1)
        elif lossf == "huber":
            self.loss_fun = torch.nn.HuberLoss(reduction="mean", delta=0.1)
        elif lossf == "mse":
            self.loss_fun = torch.nn.MSELoss()
        else:
            self.loss_fun = torch.nn.L1Loss()

    def forward(self, pred_x, pred_q, target_x, target_q):
        loss_x = self.loss_fun(pred_x.float(), target_x.float())
        loss_q = self.loss_fun(pred_q.float(), target_q.float())

        loss = torch.exp(-self.sx) * loss_x \
               + self.sx \
               + torch.exp(-self.sq) * loss_q \
               + self.sq

        self.loss_print = [loss.item(), loss_x.item(), loss_q.item()]

        return loss, loss_x.item(), loss_q.item()


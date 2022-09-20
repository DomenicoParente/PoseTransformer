import torch
import modules
from modules import *
from einops import rearrange, reduce, repeat


class PoseTransformer(nn.Module):
    """Reference: https://github.com/mx-mark/VideoTransformer-pytorch"""
    """Model with Divided Space Time Transformer Encoder"""

    def __init__(self, num_frames, height, width, patch_time, patch_height, patch_width, channels, dim_out, ldrop=0.0,
                 embed_dims=768, num_heads=12, num_transformer_layers=12, dropout_p=0.0, norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()
        num_frames = num_frames // patch_time
        self.num_frames = num_frames
        self.embed_dims = embed_dims
        self.num_transformer_layers = num_transformer_layers
        self.patch_time = patch_time
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.num_time_transformer_layers = 0

        # tokenize & position embedding
        self.patch_embed = PatchEmbed(width, height, patch_width, patch_height, patch_time, channels, embed_dims)
        num_patches = self.patch_embed.num_patches

        transformer_layers = nn.ModuleList([])
        self.num_time_transformer_layers = 4

        spatial_transformer = TransformerContainer(
            num_transformer_layers=num_transformer_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_frames=num_frames,
            norm_layer=norm_layer,
            hidden_channels=embed_dims * 4,
            operator_order=['self_attn', 'ffn'])

        temporal_transformer = TransformerContainer(
            num_transformer_layers=self.num_time_transformer_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_frames=num_frames,
            norm_layer=norm_layer,
            hidden_channels=embed_dims * 4,
            operator_order=['self_attn', 'ffn'])

        transformer_layers.append(spatial_transformer)
        transformer_layers.append(temporal_transformer)

        self.transformer_layers = transformer_layers
        self.norm = norm_layer(embed_dims, eps=1e-6)

        self.token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        # whether to add one cls_token in temporal pos_enb
        num_frames = num_frames + 1
        num_patches = num_patches + 1

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dims))
        self.drop_after_pos = nn.Dropout(p=dropout_p)
        self.drop_after_time = nn.Dropout(p=dropout_p)

        self.pos = modules.PosLinear(dim=self.embed_dims, t=num_frames - 1, dropout=ldrop)
        self.ori = modules.OriLinear(dim=self.embed_dims, t=num_frames - 1, dropout=ldrop)

        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.time_embed, std=.02)

    def prepare_tokens(self, x):
        # Tokenize
        batch = x.shape[0]
        x = self.patch_embed(x)
        # Add Position Embedding
        tokens = repeat(self.token, 'b ... -> (repeat b) ...', repeat=x.shape[0])
        x = torch.cat((tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        return x, tokens, batch

    def forward(self, x):
        x, tokens, b = self.prepare_tokens(x)
        # fact encoder - CRNN style
        spatial_transformer, temporal_transformer, = *self.transformer_layers,
        x = spatial_transformer(x)
        # Add Time Embedding
        tokens = x[:b, 0, :].unsqueeze(1)
        x = rearrange(x[:, 1:, :], '(b t) p d -> b t p d', b=b)
        x = reduce(x, 'b t p d -> b t d', 'mean')
        x = torch.cat((tokens, x), dim=1)
        x = x + self.time_embed
        x = self.drop_after_time(x)
        x = temporal_transformer(x)
        x = self.norm(x)
        x = x[:, 1:]
        pos = self.pos(x)
        ori = self.ori(x)
        return pos, ori

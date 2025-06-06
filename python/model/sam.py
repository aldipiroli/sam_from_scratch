import math

import torch
import torch.nn as nn
from torchvision.models import vit_b_16


class ViTB16(nn.Module):
    def __init__(self):
        super(ViTB16, self).__init__()
        self.model = vit_b_16(weights="ViT_B_16_Weights.IMAGENET1K_V1")
        self.model.eval()

    def forward(self, x):
        # extract feature layer: https://github.com/pytorch/vision/blob/13ada5645a4ca31242c53b3525616e43cb9ba2c7/torchvision/models/vision_transformer.py#L289
        x = self.model._process_input(x)
        n = x.shape[0]
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        # x = self.model.encoder(x)  # (b,num_patches,embeds) -> (b, 197, 768)
        x = self.model.encoder(x)[:, 1:, :]  # (b,num_patches,embeds) -> (b, 196, 768) (no cls token)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, frozen=True, target_embed_size=256):
        super(ImageEncoder, self).__init__()
        self.encoder = ViTB16()
        self.frozen = frozen
        self.original_embed_size = 768
        self.encoder.eval()
        self.embed_proj = nn.Linear(self.original_embed_size, target_embed_size)

    def forward(self, x):
        if self.frozen:
            self.encoder.eval()
            with torch.no_grad():
                embed = self.encoder(x)
        else:
            self.encoder.train()
            embed = self.encoder(x)
        embed = self.embed_proj(embed)
        return embed


class FourierPositionalEncodings(nn.Module):
    def __init__(self, num_frequencies=1):
        super(FourierPositionalEncodings, self).__init__()
        # https://proceedings.neurips.cc/paper_files/paper/2020/file/55053683268957697aa39fba6f231c68-Paper.pdf
        self.num_frequencies = num_frequencies
        self.frequencies = torch.tensor([2**i * torch.pi for i in range(self.num_frequencies)])

    def forward(self, x):
        b, num_promprs, prompt_dim = x.shape
        x = x.unsqueeze(-1) * self.frequencies.to(x.device)
        sin = torch.sin(x)
        cos = torch.cos(x)
        x_pos_encode = torch.cat([sin, cos], -1)
        x_pos_encode = x_pos_encode.reshape(b, num_promprs, -1)
        return x_pos_encode


def get_fixed_sin_positional_encodings(batch_size, num_patches, embed_size):
    pos_encodings = torch.zeros(num_patches, embed_size)
    for pos in range(num_patches):
        for i in range(embed_size):
            if i % 2 == 0:
                pos_encodings[pos, i] = math.sin(pos / 10000 ** (2 * i / embed_size))
            else:
                pos_encodings[pos, i] = math.cos(pos / 10000 ** (2 * i / embed_size))
    pos_encodings = pos_encodings.unsqueeze(0).expand(batch_size, num_patches, embed_size)
    return pos_encodings


def add_positional_embeddings(img_embeddings, pos_embeddings):
    return img_embeddings + pos_embeddings


class PointPromptEconder(nn.Module):
    def __init__(self, embed_size=256, num_frequencies=4):
        super(PointPromptEconder, self).__init__()
        self.size_pos_encode = 2 * 2 * num_frequencies
        self.embed_size = embed_size

        self.fourier_pos_encode = FourierPositionalEncodings(num_frequencies)
        # self.type_embedding = nn.parameter.Parameter(data=torch.zeros(1, 1, self.embed_size))  # not quite sure
        self.embed_projection = nn.Linear(self.size_pos_encode, embed_size)

    def forward(self, x):
        x_pos_encode = self.fourier_pos_encode(x)
        x_embed = self.embed_projection(x_pos_encode)
        # x_embed = x_embed + self.type_embedding
        return x_embed


class SelfAttention(nn.Module):
    def __init__(self, input_size, out_size):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.project_k = nn.Linear(self.input_size, self.out_size)
        self.project_q = nn.Linear(self.input_size, self.out_size)
        self.project_v = nn.Linear(self.input_size, self.out_size)

    def forward(self, in_k, in_q, in_v):
        k = self.project_k(in_k)
        q = self.project_q(in_q)
        v = self.project_v(in_v)

        qk = q @ k.transpose(-2, -1) / (self.out_size**0.5)
        qk = torch.nn.functional.softmax(qk, -1)
        attention = qk @ v
        return attention


class MultiheadAttention(nn.Module):
    def __init__(self, input_size=256, out_size=256, num_heads=2):
        super(MultiheadAttention, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.num_heads = num_heads
        head_out_size = int(out_size / num_heads)
        assert input_size % num_heads == 0

        self.attention_heads = torch.nn.ModuleList([SelfAttention(input_size, head_out_size) for _ in range(num_heads)])
        self.lin_proj = nn.Linear(self.out_size, self.out_size)

    def forward(self, in_k, in_q, in_v):
        head_outputs = []
        for attention_head in self.attention_heads:
            head_outputs.append(attention_head(in_k, in_q, in_v))
        head_outputs = torch.cat(head_outputs, -1)
        head_outputs = self.lin_proj(head_outputs)
        return head_outputs


class MaskDecoderLayer(nn.Module):
    def __init__(self, embed_size=256, dropout=0.1):
        super(MaskDecoderLayer, self).__init__()
        self.embed_size = embed_size

        self.token_self_attn = MultiheadAttention(embed_size, embed_size)
        self.token_to_img_corss_attn = MultiheadAttention(embed_size, embed_size)
        self.img_to_token_cross_attn = MultiheadAttention(embed_size, embed_size)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, embed_size),
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, img_embed):
        b, n, d = img_embed.shape
        fixed_pos_encodings = get_fixed_sin_positional_encodings(batch_size=b, num_patches=n, embed_size=d).to(
            tokens.device
        )
        # Self-attention on tokens
        tokens_ = self.token_self_attn(in_k=tokens, in_q=tokens, in_v=tokens)
        tokens = tokens + self.dropout(tokens_)
        tokens = self.norm1(tokens)

        # Cross-attention from tokens to image
        img_with_pos = add_positional_embeddings(img_embed, fixed_pos_encodings)
        token_to_img = self.token_to_img_corss_attn(in_k=img_with_pos, in_q=tokens, in_v=img_with_pos)
        tokens = tokens + self.dropout(token_to_img)
        tokens = self.norm2(tokens)

        # MLP on tokens
        tokens_ = self.mlp(tokens)
        tokens = tokens + self.dropout(tokens_)
        tokens = self.norm3(tokens)

        # Cross-attention from image to tokens
        img_to_token = self.img_to_token_cross_attn(in_k=tokens, in_q=img_with_pos, in_v=tokens)
        return tokens, img_to_token


class MaskDecoder(nn.Module):
    def __init__(self, num_decoder_layers=2, embed_size=256, dropout=0.1, num_output_masks=3, num_output_tokens=4):
        super(MaskDecoder, self).__init__()
        self.resulting_patch_size = 14  # h//patch_size (based on ViTB16)
        self.embed_size = embed_size
        self.num_output_masks = num_output_masks
        self.num_output_tokens = num_output_tokens
        self.decoder_layers = torch.nn.ModuleList(
            [MaskDecoderLayer(embed_size=embed_size, dropout=dropout) for _ in range(num_decoder_layers)]
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_size, embed_size, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_size, embed_size, kernel_size=2, stride=2),
            nn.GELU(),
        )  # upsample 4x
        self.token_to_img_corss_attn = MultiheadAttention(embed_size, embed_size)
        self.mlp_mask = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, embed_size),
        )
        self.mask_finteuer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

        self.mlp_iou = nn.Sequential(nn.Linear(embed_size, self.num_output_masks))

    def forward(self, tokens, img_embed):
        for curr_decoder in self.decoder_layers:
            tokens, img_embed = curr_decoder(tokens, img_embed)
        b, n, d = img_embed.shape

        img_embed_reshape = img_embed.permute(0, 2, 1).reshape(
            b, self.embed_size, self.resulting_patch_size, self.resulting_patch_size
        )  # (b,embed_size, h//patch_size, w//patch_size)
        img_embed_upsample = self.upsample(img_embed_reshape)
        img_embed_upsample_reshape = img_embed_upsample.reshape(
            b, self.embed_size, (self.resulting_patch_size * 4) ** 2
        )  # (b,embed_size, h//patch_size * w//patch_size)

        token_to_img_res = self.token_to_img_corss_attn(
            in_k=img_embed,
            in_q=tokens,
            in_v=img_embed,
        )
        token_to_img_res += tokens
        mask_token = token_to_img_res[:, : self.num_output_masks, :]
        mask_token = self.mlp_mask(mask_token)
        masks = torch.matmul(mask_token, img_embed_upsample_reshape)
        masks_reshape = masks.reshape(
            b, self.num_output_masks, (self.resulting_patch_size * 4), (self.resulting_patch_size * 4)
        )
        masks_reshape = self.mask_finteuer(masks_reshape)
        iou_token = token_to_img_res[:, self.num_output_tokens, :]
        iou = self.mlp_iou(iou_token)

        # TODO: handle single/multi prompt cases
        return masks_reshape, iou


class SimpleMaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 56 * 56)
        self.final_conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        x = x.mean(dim=0, keepdim=True)
        x = self.fc(x)
        x = x.mean(dim=1)
        x = x.view(1, 1, 56, 56)
        return x


class SAM(nn.Module):
    def __init__(
        self,
        embed_size=256,
        num_output_masks=1,
        num_decoder_layers=1,
        num_frequencies=1,
        dropout=0.1,
    ):
        super(SAM, self).__init__()
        self.embed_size = embed_size
        self.num_output_masks = num_output_masks
        self.num_output_tokens = num_output_masks + 1  # iou token
        self.image_encoder = ImageEncoder(frozen=True, target_embed_size=embed_size)
        self.prompt_encoder = PointPromptEconder(embed_size=embed_size, num_frequencies=num_frequencies)
        self.output_token = nn.parameter.Parameter(data=torch.zeros(1, self.num_output_tokens, 1))

        self.mask_decoder = MaskDecoder(
            num_decoder_layers=num_decoder_layers,
            embed_size=embed_size,
            dropout=dropout,
            num_output_tokens=self.num_output_tokens,
            num_output_masks=self.num_output_masks,
        )

    def forward(self, img, prompt):
        # img embedding
        img_embed = self.image_encoder(img)
        b, n, d = img_embed.shape

        # prompt token embedd + output token
        prompt_tokens = self.prompt_encoder(prompt)
        output_token = self.output_token.expand(b, self.num_output_tokens, self.embed_size)
        tokens = torch.cat([output_token, prompt_tokens], 1)

        # mask decoder
        masks, iou = self.mask_decoder(tokens, img_embed)
        return masks, iou


if __name__ == "__main__":
    cfg = {}
    model = SAM(cfg)

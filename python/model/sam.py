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
        x = self.model.encoder(x)  # (b,num_patches,embeds) -> (b, 197, 768)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, frozen=True, original_embed_size=768, target_embed_size=256):
        super(ImageEncoder, self).__init__()
        self.encoder = ViTB16()
        self.frozen = frozen
        self.encoder.eval()
        self.embed_proj = nn.Linear(original_embed_size, target_embed_size)

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
    def __init__(self, num_frequencies=4):
        super(FourierPositionalEncodings, self).__init__()
        # https://proceedings.neurips.cc/paper_files/paper/2020/file/55053683268957697aa39fba6f231c68-Paper.pdf
        self.num_frequencies = num_frequencies
        self.frequencies = torch.tensor([2**i * torch.pi for i in range(self.num_frequencies)])

    def forward(self, x):
        b = x.shape[0]
        x = x.unsqueeze(-1) * self.frequencies
        sin = torch.sin(x)
        cos = torch.cos(x)
        x_pos_encode = torch.cat([sin, cos], -1)
        x_pos_encode = x_pos_encode.reshape(b, -1)
        return x_pos_encode


class PointPromptEconder(nn.Module):
    def __init__(self, embed_size=256, num_frequencies=4):
        super(PointPromptEconder, self).__init__()
        self.size_pos_encode = 2 * 2 * num_frequencies
        self.embed_size = embed_size

        self.fourier_pos_encode = FourierPositionalEncodings(num_frequencies)
        self.type_embedding = nn.parameter.Parameter(data=torch.zeros(1))
        self.embed_projection = nn.Linear(self.size_pos_encode, embed_size)

    def forward(self, x):
        x_pos_encode = self.fourier_pos_encode(x)
        x_embed = self.embed_projection(x_pos_encode)
        x_embed = x_embed + self.type_embedding
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
        q = self.project_k(in_q)
        v = self.project_k(in_v)

        qk = q @ k.transpose(-2, -1) * (self.out_size**0.5)
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
        assert input_size % input_size == 0

        self.attention_heads = [SelfAttention(input_size, head_out_size) for _ in range(num_heads)]
        self.lin_proj = nn.Linear(self.out_size, self.out_size)

    def forward(self, in_k, in_q, in_v):
        head_outputs = []
        for attention_head in self.attention_heads:
            head_outputs.append(attention_head(in_k, in_q, in_v))
        head_outputs = torch.cat(head_outputs, -1)
        head_outputs = self.lin_proj(head_outputs)
        return head_outputs


class MaskDecoderLayer(nn.Module):
    def __init__(self):
        super(MaskDecoderLayer, self).__init__()

    def forward(self, x):
        return x


class MaskDecoder(nn.Module):
    def __init__(self, embed_size=256):
        super(MaskDecoder, self).__init__()
        self.embed_size = embed_size
        self.token_self_attention = MultiheadAttention(input_size=embed_size, out_size=embed_size, num_heads=2)

    def forward(self, prompt_tokens, img_embed):
        b, n, d = prompt_tokens.shape
        tokens = prompt_tokens
        token_sa = self.token_self_attention(tokens, tokens, tokens)
        return tokens


###########################################
import debugpy

debugpy.listen(("localhost", 6001))
print("Waiting for debugger attach...")
debugpy.wait_for_client()


###########################################
class SAM(nn.Module):
    def __init__(self, embed_size=256):
        super(SAM, self).__init__()
        self.image_encoder = ImageEncoder(frozen=True)
        self.prompt_encoder = PointPromptEconder(embed_size=embed_size)
        self.output_token = nn.parameter.Parameter(data=torch.zeros(1))
        self.proj_prompt_tokens = nn.Linear(embed_size + 1, embed_size)

    def forward(self, img, prompt):
        img_embed = self.image_encoder(img)
        b, n, d = img_embed.shape

        prompt_tokens = self.prompt_encoder(prompt)
        output_token = self.output_token.unsqueeze(-1).unsqueeze(-1).expand(b, n, 1)
        tokens = torch.cat([output_token, prompt_tokens], -1)
        tokens = self.proj_prompt_tokens(tokens)

        return {}


if __name__ == "__main__":
    cfg = {}
    model = SAM(cfg)
    x = torch.randn(8, 3, 224, 224)
    out = model(x)

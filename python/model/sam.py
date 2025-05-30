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
        x = self.model.encoder(x)  # (b,num_patches,embeds)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, frozen=True):
        super(ImageEncoder, self).__init__()
        self.encoder = ViTB16()
        self.frozen = frozen
        self.encoder.eval()

    def forward(self, x):
        if self.frozen:
            self.encoder.eval()
            with torch.no_grad():
                out = self.encoder(x)
        else:
            self.encoder.train()
            out = self.encoder(x)
        return out


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
        size_pos_encode = 2 * 2 * num_frequencies
        self.embed_size = embed_size

        self.fourier_pos_encode = FourierPositionalEncodings(num_frequencies)
        self.type_embedding = nn.parameter.Parameter(data=torch.zeros(1))
        self.embed_projection = nn.Linear(size_pos_encode, embed_size)

    def forward(self, x):
        x_pos_encode = self.fourier_pos_encode(x)
        x_embed = self.embed_projection(x_pos_encode)
        x_embed = x_embed + self.type_embedding
        return x_embed


class MaskDecoder(nn.Module):
    def __init__(self):
        super(MaskDecoder, self).__init__()

    def forward(self, x):
        return x


class SAM(nn.Module):
    def __init__(self, cfg):
        super(SAM, self).__init__()
        self.cfg = cfg
        self.image_encoder = ImageEncoder(frozen=True)

    def forward(self, x):
        y = self.image_encoder(x)
        return_dict = {"x": x, "y": y}
        return return_dict


if __name__ == "__main__":
    cfg = {}
    model = SAM(cfg)
    x = torch.randn(8, 3, 224, 224)
    out = model(x)

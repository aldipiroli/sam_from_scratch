import torch
import torch.nn as nn
from torchvision.models import vit_b_16

class ViTB16(nn.Module):
    def __init__(self):
        super(ViTB16, self).__init__()
        self.model = vit_b_16(weights="ViT_B_16_Weights.IMAGENET1K_V1")
        self.model.eval()

    def forward(self, x):
        # extract feature layer: https://github.com/pytorch/vision/blob/13ada5645a4ca31242c53b3525616e43cb9ba2c7/torchvision/models/vision_transformer.py#L289
        x = self.model._process_input(x)
        n = x.shape[0]
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.model.encoder(x) # (b,num_patches,embeds) 
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

class SAM(nn.Module):
    def __init__(self, cfg):
        super(SAM, self).__init__()
        self.cfg = cfg
        self.image_encoder = ImageEncoder(frozen=True) 

    def forward(self, x):
        y = self.image_encoder(x)
        return_dict = {"x": x, "y":y}
        return return_dict

if __name__ == "__main__":
    cfg = {}
    model = SAM(cfg)
    x = torch.randn(8, 3, 224, 224)
    out = model(x)
    print(f"x {out["x"].shape}")
    print(f"y {out["y"].shape}")

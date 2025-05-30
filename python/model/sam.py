import torch
import torch.nn as nn


class SAM(nn.Module):
    def __init__(self, cfg):
        super(SAM, self).__init__()
        self.cfg = cfg

    def forward(self, x):
        y = x
        return_dict = {"x": x, "y":y}
        return return_dict

if __name__ == "__main__":
    cfg = {}
    model = SAM(cfg)
    x = torch.randn(8, 3, 256, 256)
    out = model(x)
    print(f"x {out["x"].shape}")
    print(f"y {out["y"].shape}")

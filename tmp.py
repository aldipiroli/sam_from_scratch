from torchvision.models import vit_b_16

# Load the pretrained ViT-Base model with 16x16 patches from torchvision
model = vit_b_16(pretrained=True)
model.eval()

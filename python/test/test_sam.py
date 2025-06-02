import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.sam import (
    SAM,
    FourierPositionalEncodings,
    ImageEncoder,
    MaskDecoder,
    MaskDecoderLayer,
    MultiheadAttention,
    PointPromptEconder,
    SelfAttention,
    get_fixed_sin_positional_encodings,
)


def test_fourier_positional_encoding():
    num_frequencies = 4
    b = 2
    num_promprs = 2
    pos_encoder = FourierPositionalEncodings(num_frequencies)
    x = torch.randn(b, num_promprs, 2, requires_grad=True)
    x_encode = pos_encoder(x)
    assert x_encode.shape == (b, num_promprs, 2 * 2 * num_frequencies)
    assert x_encode.requires_grad


def test_fixed_sinusoidal_positional_encodings():
    num_patches = 64
    embed_size = 256
    batch_size = 2
    pos_encodings = get_fixed_sin_positional_encodings(
        batch_size=batch_size, num_patches=num_patches, embed_size=embed_size
    )
    assert pos_encodings.shape == (batch_size, num_patches, embed_size)
    assert not torch.isnan(pos_encodings).any()


def test_point_prompt_encoder():
    b = 2
    embed_size = 256
    num_promprs = 2
    point_prompt_encoder = PointPromptEconder(embed_size=embed_size)
    x = torch.randn(b, num_promprs, 2, requires_grad=True)
    x_encode = point_prompt_encoder(x)
    assert x_encode.shape == (b, num_promprs, embed_size)
    assert x_encode.requires_grad


def test_self_attention():
    b = 2
    n = 64
    d = 256
    self_attention = SelfAttention(input_size=256, out_size=256)
    x = torch.randn(b, n, d, requires_grad=True)
    y = self_attention(x, x, x)
    assert y.shape == x.shape
    assert y.requires_grad


def test_multihead_attention():
    b = 2
    n = 64
    d = 256
    multihead_attention = MultiheadAttention(input_size=256, out_size=256, num_heads=2)
    x = torch.randn(b, n, d, requires_grad=True)
    y = multihead_attention(x, x, x)
    assert y.shape == x.shape
    assert y.requires_grad


def test_image_encoder():
    target_embed_size = 256
    img_encoder = ImageEncoder(target_embed_size=target_embed_size)
    b, c, h, w = 2, 3, 224, 224
    x = torch.randn(b, c, h, w, requires_grad=True)
    x_encoded = img_encoder(x)
    assert x_encoded.shape[-1] == target_embed_size
    assert x_encoded.requires_grad


def test_mask_decoder_layer():
    b = 2
    n = 196
    d = 256
    num_prompts = 1
    mask_decoder_layer = MaskDecoderLayer(embed_size=256, dropout=0.1)

    tokens = torch.randn(b, num_prompts, d, requires_grad=True)
    img_embed = torch.randn(b, n, d, requires_grad=True)
    tokens_, img_embed_ = mask_decoder_layer(tokens, img_embed)
    assert tokens.shape == tokens_.shape
    assert img_embed.shape == img_embed_.shape
    assert tokens_.requires_grad
    assert img_embed_.requires_grad


def test_mask_decoder():
    b = 2
    n = 196
    d = 256
    num_prompts = 1
    resulting_patch_size = 14
    upscale_factor = 4
    num_output_masks = 3
    num_iou_tokens = 1
    num_output_tokens = num_output_masks + num_iou_tokens
    mask_decoder = MaskDecoder(
        num_decoder_layers=2,
        embed_size=256,
        dropout=0.1,
        num_output_masks=num_output_masks,
        num_output_tokens=num_output_tokens,
    )

    tokens = torch.randn(b, num_prompts + 4, d, requires_grad=True)
    img_embed = torch.randn(b, n, d, requires_grad=True)
    masks, iou = mask_decoder(tokens, img_embed)
    assert masks.shape == (
        b,
        num_output_masks,
        (resulting_patch_size * upscale_factor),
        (resulting_patch_size * upscale_factor),
    )
    assert iou.shape == (b, num_output_masks)
    assert masks.requires_grad
    assert iou.requires_grad


def test_sam():
    num_output_masks = 3
    sam = SAM(num_output_masks=num_output_masks)
    b, c, h, w = 2, 3, 224, 224
    num_prompts = 1
    resulting_patch_size = 14
    upscale_factor = 4
    img = torch.randn(b, c, h, w, requires_grad=True)
    prompt = torch.randint(0, h, size=(b, num_prompts, 2)).float()
    prompt.requires_grad = True
    masks, iou = sam(img, prompt)
    assert masks.shape == (
        b,
        num_output_masks,
        (resulting_patch_size * upscale_factor),
        (resulting_patch_size * upscale_factor),
    )
    assert iou.shape == (b, num_output_masks)
    assert masks.requires_grad
    assert iou.requires_grad


@pytest.mark.skip()
def test_gradient_flow(skip=True):
    if skip:
        return True
    model = SAM()
    b, c, h, w = 2, 3, 224, 224
    num_prompts = 1

    img = torch.randn(b, c, h, w, requires_grad=True)
    gt_mask = torch.randn(b, 4, 56, 56, requires_grad=False)
    gt_iou = torch.randn(b, 4)
    prompt = torch.randint(0, h, size=(b, num_prompts, 2)).float()
    prompt.requires_grad = True

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    optimizer.zero_grad()
    masks, iou = model(img, prompt)
    loss_mask = criterion(masks.reshape(-1), gt_mask.reshape(-1))
    loss_iou = criterion(iou.reshape(-1), gt_iou.reshape(-1))
    loss = loss_mask + loss_iou
    loss.backward()

    excluded = ["image_encoder.encoder.model"]
    flow_problem = False
    for name, param in model.named_parameters():
        if any(exclude in name for exclude in excluded):
            continue
        if param.grad is None:
            print(f"Parameter '{name}' has NO gradient!")
            flow_problem = True
    assert not flow_problem


if __name__ == "__main__":
    test_self_attention()
    test_multihead_attention()
    test_fourier_positional_encoding()
    test_fixed_sinusoidal_positional_encodings()
    test_point_prompt_encoder()
    test_mask_decoder_layer()
    test_mask_decoder()
    test_image_encoder()
    test_sam()
    test_gradient_flow()
    print("Tests passed!")

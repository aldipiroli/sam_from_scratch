import os
import sys

import torch

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
    x = torch.randn(b, num_promprs, 2)
    x_encode = pos_encoder(x)
    assert x_encode.shape == (b, num_promprs, 2 * 2 * num_frequencies)


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
    x = torch.randn(b, num_promprs, 2)
    x_encode = point_prompt_encoder(x)
    assert x_encode.shape == (b, num_promprs, embed_size)


def test_self_attention():
    b = 2
    n = 64
    d = 256
    self_attention = SelfAttention(input_size=256, out_size=256)

    x = torch.randn(b, n, d)
    y = self_attention(x, x, x)
    assert y.shape == x.shape


def test_multihead_attention():
    b = 2
    n = 64
    d = 256
    multihead_attention = MultiheadAttention(input_size=256, out_size=256, num_heads=2)

    x = torch.randn(b, n, d)
    y = multihead_attention(x, x, x)
    assert y.shape == x.shape


def test_image_encoder():
    original_embed_size = 768
    target_embed_size = 256
    img_encoder = ImageEncoder(original_embed_size=original_embed_size, target_embed_size=target_embed_size)

    b, c, h, w = 2, 3, 224, 224
    x = torch.randn(b, c, h, w)
    x_encoded = img_encoder(x)
    assert x_encoded.shape[-1] == target_embed_size


def test_mask_decoder_layer():
    b = 2
    n = 196
    d = 256
    num_prompts = 2
    mask_decoder_layer = MaskDecoderLayer(embed_size=256, dropout=0.1)

    tokens = torch.randn(b, num_prompts, d)
    img_embed = torch.randn(b, n, d)
    tokens_, img_embed_ = mask_decoder_layer(tokens, img_embed)
    assert tokens.shape == tokens_.shape
    assert img_embed.shape == img_embed_.shape


def test_mask_decoder():
    b = 2
    n = 196
    d = 256
    num_prompts = 2
    resulting_patch_size = 14
    upscale_factor = 4
    num_output_tokens = 4
    mask_decoder = MaskDecoder(
        num_decoder_layers=2, embed_size=256, dropout=0.1, resulting_patch_size=resulting_patch_size
    )

    tokens = torch.randn(b, num_prompts + 4, d)
    img_embed = torch.randn(b, n, d)
    masks, iou = mask_decoder(tokens, img_embed)
    assert masks.shape == (
        b,
        num_output_tokens,
        (resulting_patch_size * upscale_factor),
        (resulting_patch_size * upscale_factor),
    )
    assert iou.shape == (b, num_output_tokens)


def test_sam():
    sam = SAM()
    b, c, h, w = 2, 3, 224, 224
    num_prompts = 2
    resulting_patch_size = 14
    upscale_factor = 4
    num_output_tokens = 4
    img = torch.randn(b, c, h, w)
    prompt = torch.randint(0, h, size=(b, num_prompts, 2))
    masks, iou = sam(img, prompt)
    assert masks.shape == (
        b,
        num_output_tokens,
        (resulting_patch_size * upscale_factor),
        (resulting_patch_size * upscale_factor),
    )
    assert iou.shape == (b, num_output_tokens)


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
    print("Tests passed!")

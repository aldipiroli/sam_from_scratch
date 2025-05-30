import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.sam import (
    SAM,
    FourierPositionalEncodings,
    ImageEncoder,
    MaskDecoder,
    MultiheadAttention,
    PointPromptEconder,
    SelfAttention,
)


def test_fourier_positional_encoding():
    num_frequencies = 4
    b = 2
    pos_encoder = FourierPositionalEncodings(num_frequencies)
    x = torch.randn(b, 2)
    x_encode = pos_encoder(x)
    assert x_encode.shape == (b, 2 * 2 * num_frequencies)


def test_point_prompt_encoder():
    b = 2
    embed_size = 256
    point_prompt_encoder = PointPromptEconder(embed_size=embed_size)
    x = torch.randn(b, 2)
    x_encode = point_prompt_encoder(x)
    assert x_encode.shape == (b, embed_size)


def test_mask_decoder():
    b = 2
    n = 64
    d = 256
    mask_decoder = MaskDecoder()

    prompt_tokens = torch.randn(b, n, d)
    img_embed = torch.randn(b, 197, d)
    x_encode = mask_decoder(prompt_tokens, img_embed)


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


def test_sam():
    sam = SAM()
    b, c, h, w = 2, 3, 224, 224
    img = torch.randn(b, c, h, w)
    prompt = torch.randint(0, h, size=(b, 2))
    y = sam(img, prompt)


def test_image_encoder():
    original_embed_size = 768
    target_embed_size = 256
    img_encoder = ImageEncoder(original_embed_size=original_embed_size, target_embed_size=target_embed_size)

    b, c, h, w = 2, 3, 224, 224
    x = torch.randn(b, c, h, w)
    x_encoded = img_encoder(x)
    assert x_encoded.shape[-1] == target_embed_size


if __name__ == "__main__":
    test_fourier_positional_encoding()
    test_point_prompt_encoder()
    test_mask_decoder()
    test_self_attention()
    test_multihead_attention()
    test_sam()
    test_image_encoder()
    print("Tests passed!")

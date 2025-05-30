import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.sam import FourierPositionalEncodings, PointPromptEconder


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


if __name__ == "__main__":
    test_fourier_positional_encoding()
    test_point_prompt_encoder()
    print("Tests passed!")

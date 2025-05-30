import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.sam import FourierPositionalEncodings

def test_fourier_positional_encoding():
    num_frequencies = 4
    b = 2
    pos_encoder = FourierPositionalEncodings(num_frequencies)
    x = torch.randn(b, 2)
    x_encode = pos_encoder(x)
    assert x_encode.shape == (b, 2 * 2 * num_frequencies)


if __name__ == "__main__":
    test_fourier_positional_encoding()
    print("Tests passed!")

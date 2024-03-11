import mlx.core as mx
import mlx.nn as nn

import math

"""
Credit to: https://github.com/kyegomez/BitNet/blob/main/bitnet/bitlinear.py
"""
class BitLinear(nn.Linear):
    def __init__(self,
                 input_dims: int,
                 output_dims: int, 
                 bias: bool = True, 
                 num_groups: int = 1, 
                 bits: int = 2) -> None:
        super().__init__(input_dims, output_dims, bias)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_groups = num_groups
        self.eps = 1e-5
        self.norm = nn.LayerNorm(input_dims)

        # Quantiziation and dequantization
        self.Q_b = 2 ** (bits - 1)
        self.beta = mx.zeros((self.weight.shape[0],))
        self.gamma = mx.zeros((self.weight.shape[0],))

    def ste(self, x):
        binarized_x = mx.sign(x)
        binarized_x = (binarized_x - x) + x
        return binarized_x

    def binarize_weights_groupwise(self):
        group_size = self.weight.shape[0] // self.num_groups
        binarized_weights = mx.zeros_like(self.weight)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = self.weight[start_idx:end_idx]
            alpha_g = weight_group.mean()
            self.beta[start_idx:end_idx] = weight_group.abs().mean()
            binarized_weights[start_idx:end_idx] = self.ste(weight_group - alpha_g)

        return binarized_weights

    def quantize_activations_groupwise(self, x):
        group_size = x.shape[0] // self.num_groups
        quantized_x = mx.zeros_like(x)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            activation_group = x[start_idx:end_idx]

            gamma_g = activation_group.abs().max()
            self.gamma[start_idx:end_idx] = gamma_g
            quantized_x[start_idx:end_idx] = mx.clip(
                activation_group * self.Q_b / (gamma_g + self.eps),
                -self.Q_b + self.eps,
                self.Q_b - self.eps,
            )

        return quantized_x

    def dequantize_activations_groupwise(self, x):
        return x * self.gamma * self.beta / self.Q_b

    def __call__(self, x: mx.array) -> mx.array:
        # Normalize input
        x = self.norm(x)

        # Binarize weights and quantize activations
        binarized_weights = self.binarize_weights_groupwise()

        # Quantize input
        x_quant = self.quantize_activations_groupwise(x)

        # Perform linear transformation
        if "bias" in self:
            output = mx.addmm(self.bias, x_quant, binarized_weights.transpose())
        else:
            output = x_quant @ binarized_weights.transpose()

        # Dequantize activations
        output = self.dequantize_activations_groupwise(output)

        # Return output
        return output


def test_bitlinear_initialization():
    bitlinear = BitLinear(input_dims=512, output_dims=256, bias=True)
    assert bitlinear.input_dims == 512
    assert bitlinear.output_dims == 256
    assert bitlinear.weight.shape == (256, 512)
    assert bitlinear.bias.shape == (256,)
    assert bitlinear.gamma.shape == (256,)
    assert bitlinear.beta.shape == (256,)


def test_bitlinear_forward_pass():
    bitlinear = BitLinear(input_dims=512, output_dims=256, bias=True)
    x = mx.random.normal([1, 512])
    out = bitlinear(x)
    assert out.shape == (1, 256)


def test_bitlinear_no_bias():
    bitlinear = BitLinear(input_dims=512, output_dims=256, bias=False)
    assert "bias" not in bitlinear


def test_bitlinear_quantization():
    bitlinear = BitLinear(input_dims=512, output_dims=256, bias=True)
    x = mx.random.uniform(shape=[1, 512])
    out = bitlinear(x)
    out = mx.round(out)
    assert mx.all(out <= bitlinear.beta.reshape(out.shape))

if __name__ == "__main__":
    # Example usage
    bitlinear = BitLinear(10, 6, bits=2)
    input_tensor = mx.random.normal([6, 10])  # Example input tensor
    output = bitlinear(input_tensor)
    print(mx.round(output))  # Example output tensor
    test_bitlinear_initialization()
    test_bitlinear_forward_pass()
    test_bitlinear_no_bias()
    test_bitlinear_quantization()
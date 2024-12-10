#NEW NN
from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from typing import Optional
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height, new_width = height // kh, width // kw

    output = input.contiguous()
    output = input.view(batch, channel, new_height, kh, width, kw,)
    output = output.permute(0, 1, 2, 4, 3, 5)
    output = output.contiguous()
    output = output.view(batch, channel, new_height, new_width, kh * kw)

    return output, new_height, new_width

def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D average pooling on input with kernel kernel"""
    batch, channel = input.shape[:2]
    input, new_height, new_width = tile(input, kernel)
    out = input.mean(-1)
    return out.view(batch, channel, new_height, new_width)

max_reduce = FastOps.reduce(operators.max, float("-inf"))
''' class Max(Function):
    """A function to compute the maximum value along a specified dimension."""
    
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Computes the forward pass of sum."""
        ctx.save_for_backward(a, int(dim.item()))  # use a mask in order to retrieve it
        return max_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass of sum."""
        a, dim = ctx.saved_values
        one_hot = argmax(a, int(dim))
        return grad_output * one_hot, tensor([0.0]) '''

def max(input: Tensor, dim: int) -> Tensor:
    """Reduce max of input tensor on a specified dimension."""
    max_val = max_reduce(input, dim)
    out = list(input.shape)
    out[dim] = 1
    max_shape = max_val.view(*out)
    return max_shape


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor."""
    return max_reduce(input, dim) == input


def softmax(input: Tensor, dim: int) -> Tensor:
    """Take the softmax of an input tensor along the specified dimension."""
    expInput = input.exp()
    return expInput / expInput.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax along the specified dimension."""
    val = max(input, dim) 
    exp_sum = ((input - val).exp()).sum(dim).log()
    return input - exp_sum - val



def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Do 2D max pooling using the specified kernel."""
    output, new_height, new_width = tile(input, kernel)
    pooled = max(output, dim=4) 
    return pooled.view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor with probability rate."""
    if ignore:
        return input
    else:
        prob_rate = rand(input.shape)
        idx = prob_rate > rate
        return input * idx

"""
Rounding operations.
"""
import mindspore.nn as nn
from mindspore.ops import operations as ops


class RoundQuant(nn.Cell):
    """
    Rounding operation: discretize floating points to certain bins.
    """
    def __init__(self, inverse_bin_width):
        super(RoundQuant, self).__init__()
        self.inverse_bin_width = float(inverse_bin_width)
        self.round = ops.Round()

    def construct(self, x):
        """construct"""
        h = x * self.inverse_bin_width
        h = self.round(h)
        h = h / self.inverse_bin_width
        return h

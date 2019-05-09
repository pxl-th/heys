"""
Branch and Bound algorithm for finding linear approximations.
"""
from numpy import (
    arange,
    array,
    ndarray,
    zeros,
)

from heys.cipher import Heys
from heys.utilities import (
    bitwise_dot,
    calculate_hamming_weight,
)

__all__ = [
    "branch_bound",
    "sbox_linear_approximations",
]


def branch_bound(heys: Heys, probability_threshold: float):
    sbox_approximations = sbox_linear_approximations(s_box=S_BOX)

    pass


def sbox_linear_approximations(s_box: ndarray) -> ndarray:
    numbers = 1 << 4
    approximations = zeros((16, 16), dtype="int16")
    input_numbers = arange(numbers, dtype="uint16")
    hamming = calculate_hamming_weight(bits=4)

    for input_sum in range(numbers):
        for output_sum in range(numbers):
            approximations[input_sum, output_sum] = (
                bitwise_dot(
                    x=input_sum,
                    y=input_numbers,
                    hamming_weight=hamming,
                )
                ==
                bitwise_dot(
                    x=output_sum,
                    y=s_box[input_numbers],
                    hamming_weight=hamming,
                )
            ).sum()

    return (approximations - 8) / 16 + 0.5

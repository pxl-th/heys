from numpy import (
    einsum,
    ndarray,
    where,
    zeros,
)

__all__ = [
    "bitwise_dot",
    "calculate_hamming_weight",
    "linear_potential",
]


def linear_potential(
    inputs: ndarray,
    alpha: int,
    beta: int,
    transformation: callable,
    hamming_weight: ndarray,
) -> ndarray:
    inputs_number = inputs.shape[0]
    potential = bitwise_dot(x=alpha, y=inputs, hamming_weight=hamming_weight)
    potential ^= bitwise_dot(
        x=beta,
        y=transformation(inputs),
        hamming_weight=hamming_weight,
    )
    return (einsum("e->", where(potential == 0, 1, -1)) / inputs_number) ** 2


def bitwise_dot(x, y, hamming_weight: ndarray):
    return hamming_weight[x & y] & 0b1


def calculate_hamming_weight(bits: int = 16) -> ndarray:
    weight_table = zeros(1 << bits, dtype="uint16")
    for number in range(1 << bits):
        weight_table[number] = bin(number).count("1")
    return weight_table

from numpy import (
    arange,
    einsum,
    ndarray,
    where,
    zeros,
)

from heys.cipher import Heys

__all__ = [
    "bitwise_dot",
    "calculate_hamming_weight",
    "average_linear_potential",
    "linear_potential",
]


def average_linear_potential(
    inputs: ndarray,
    alpha: int,
    beta: int,
    hayes: Heys,
    hamming_weight: ndarray,
    keys_batch_size: int = 256,
) -> ndarray:
    numbers = 1 << 16
    keys_batches = numbers // keys_batch_size
    keys_batch = arange(keys_batch_size, dtype="uint16").reshape((-1, 1))
    potential_einsum, average_einsum = "rc->r", "e->"

    average_potential = 0
    for _ in range(keys_batches):
        potential = bitwise_dot(
            x=beta,
            y=hayes.permutation[hayes.sbox[inputs ^ keys_batch]],
            hamming_weight=hamming_weight,
        )
        potential ^= bitwise_dot(
            x=alpha,
            y=inputs,
            hamming_weight=hamming_weight,
        )
        potential = (
            (einsum(potential_einsum, where(potential == 0, 1, -1))
             / numbers) ** 2
        )
        average_potential += einsum(average_einsum, potential)
        keys_batch += keys_batch_size

    return average_potential / numbers


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
